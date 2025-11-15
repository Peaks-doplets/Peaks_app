import streamlit as st
import pandas as pd
import numpy as np
from nptdms import TdmsFile
from scipy.signal import find_peaks
from scipy.stats import linregress
import plotly.express as px
import plotly.graph_objects as go

# --- Inicializace Session State (Paměti aplikace) ---
# Tady se ukládají kalibrační sady, aby se data neztratila při přepínání záložek.
if 'calibration_data' not in st.session_state:
    st.session_state.calibration_data = pd.DataFrame(columns=[
        'Name', 'Value_X', 'Unit_X', 'Peak_Count', 'Avg_Height_V', 'STD_Dev_V'
    ])
if 'avg_height' not in st.session_state:
    st.session_state.avg_height = None
if 'std_dev' not in st.session_state:
    st.session_state.std_dev = None
if 'peak_count' not in st.session_state:
    st.session_state.peak_count = None


# --- Nastavení aplikace a Záhlaví ---
st.set_page_config(layout="wide", page_title="Automatická Analýza Kapičkové Mikrofluidiky")
st.title("🔬 Aplikace pro Rychlou Analýzu Fluorescenčních Píků")

# --- Struktura Záložek ---
tab1, tab2, tab3 = st.tabs(["1. Vyhodnocení záznamu", "2. Kalibrace (Lineární regrese)", "3. Nelineární regrese"])

# ------------------------------------------------------------------------------------------------
# --- ZÁLOŽKA 1: VYHODNOCENÍ ZÁZNAMU (Ukládání Dat do Kalibrace) ---
# ------------------------------------------------------------------------------------------------

with tab1:
    st.header("1. Vyhodnocení záznamu a uložení kalibračních bodů")
    
    # --- 1. Nahrání Souboru ---
    uploaded_file = st.file_uploader("Nahrajte TDMS soubor", type="tdms", key="upload_tab1")

    if uploaded_file is not None:
        
        # Načtení dat z TDMS souboru
        try:
            tdms_file = TdmsFile.read(uploaded_file)
            pmt_signal = tdms_file['Measured Data']['PMT Output (V)'].data
        except Exception as e:
            st.error(f"CHYBA: Nepodařilo se najít data 'PMT Output (V)' nebo načíst soubor. {e}")
            st.stop()
            
        data_length = len(pmt_signal)
        st.success(f"Soubor **{uploaded_file.name}** úspěšně načten. Celkem **{data_length}** datových bodů.")

        # --- 2. Interaktivní Graf a Nastavení Slicing ---
        st.subheader("Vizualizace a výběr oblasti (Slicing)")
        
        # Interaktivní graf (Plotly) pro výběr oblasti
        time_index = np.arange(data_length)
        df_signal = pd.DataFrame({'Index': time_index, 'Fluorescence': pmt_signal})
        fig_sig = px.line(df_signal, x='Index', y='Fluorescence', title='Celkový průběh Fluorescence (Zoomováním vyberte oblast)')
        fig_sig.update_traces(line=dict(width=0.5))
        st.plotly_chart(fig_sig, use_container_width=True)
        
        # Numerický vstup pro slicing
        col1, col2 = st.columns(2)
        with col1:
            start_index = st.number_input("Začátek Indexu X (Slicing)", min_value=0, max_value=data_length, value=int(data_length * 0.2))
        with col2:
            end_index = st.number_input("Konec Indexu X", min_value=0, max_value=data_length, value=int(data_length * 0.8))

        # --- 3. Nastavení Píků a Spuštění ---
        st.subheader("Detekce píků a statistika")
        
        col3, col4 = st.columns(2)
        with col3:
            min_peak_height = st.slider("Minimální výška píku (V)", 0.0, 5.0, 0.5, 0.05)
        with col4:
            min_peak_distance = st.slider("Minimální vzdálenost mezi píky (body)", 1, 1000, 700)
        
        
        # --- FUNKCE PRO VÝPOČET A ZOBRAZENÍ VÝSLEDKŮ (Definice) ---
        def analyze_and_display(pmt_signal, start, end, height, distance):
            if start >= end:
                st.error("Chyba: Začátek oblasti musí být menší než konec.")
                return None, None, None
            
            signal_to_analyze = pmt_signal[start:end]
            peaks, properties = find_peaks(signal_to_analyze, height=height, distance=distance)
            peak_heights = properties['peak_heights']
            
            # Zobrazení výsledků
            if len(peak_heights) > 0:
                avg_height = np.mean(peak_heights)
                std_dev = np.std(peak_heights)

                st.subheader(f"✅ Analýza Dokončena: Nalezeno **{len(peaks)}** Píků")
                
                # Vykreslení grafu s označenými píky
                fig_results = go.Figure()
                fig_results.add_trace(go.Scatter(y=signal_to_analyze, mode='lines', 
                                                name='Signál Fluorescence', line=dict(width=0.7, color='red')))
                fig_results.add_trace(go.Scatter(x=peaks, y=signal_to_analyze[peaks], mode='markers', 
                                                name='Nalezené Píky', marker=dict(symbol='x', size=10, color='green')))
                
                fig_results.update_layout(title=f"Detekce píků v oblasti {start} - {end}",
                                          xaxis_title=f"Index v rámci oblasti ({len(signal_to_analyze)} bodů)",
                                          yaxis_title="Intenzita Fluorescence (V)")
                
                st.plotly_chart(fig_results, use_container_width=True)
                
                # Zobrazení metrik (Průměr a STD)
                st.subheader("📊 Souhrnná Statistika")
                col_avg, col_std = st.columns(2)
                with col_avg:
                    st.metric(label="Průměrná výška píků", value=f"{avg_height:.4f} V")
                with col_std:
                    st.metric(label="Směrodatná odchylka (STD)", value=f"{std_dev:.4f} V")
                
                return avg_height, std_dev, len(peaks)
            else:
                 st.warning("Nebyly nalezeny žádné píky.")
                 return None, None, None
        
        # --- TLAČÍTKO PRO SPOUŠTĚNÍ ANALÝZY ---
        if st.button("▶️ Spustit analýzu"):
            st.session_state.avg_height, st.session_state.std_dev, st.session_state.peak_count = analyze_and_display(
                pmt_signal, start_index, end_index, min_peak_height, min_peak_distance
            )

        # --- TLAČÍTKO PRO KALIBRACI (POUZE PO ÚSPĚŠNÉ ANALÝZE) ---
        if st.session_state.avg_height is not None and st.session_state.peak_count > 0:
            st.markdown("---")
            st.subheader("Uložit pro kalibraci")
            
            # Formulář pro zadání kalibračních hodnot
            col_name, col_value, col_unit, col_button = st.columns([2, 1, 1, 1])
            with col_name:
                name_input = st.text_input("Jméno sady (např. 'Fluorescein 600nM')", 
                                            value=f"{uploaded_file.name}_Analyzed")
            with col_value:
                value_x_input = st.number_input("Hodnota X (Koncentrace)", min_value=0.0, value=1.0, key="cal_val_x")
            with col_unit:
                unit_x_input = st.selectbox("Jednotka X", ["nM", "uM", "mM", "mg/ml", "Arbitrary"], key="cal_unit_x")
            with col_button:
                st.write("") # Mezera pro zarovnání tlačítka
                
                if st.button("➕ Zahrnout do kalibrace", key="add_cal_btn"):
                    
                    # Vytvoření nového řádku dat
                    new_row = pd.DataFrame([{
                        'Name': name_input, 
                        'Value_X': value_x_input, 
                        'Unit_X': unit_x_input, 
                        'Peak_Count': st.session_state.peak_count, 
                        'Avg_Height_V': st.session_state.avg_height, 
                        'STD_Dev_V': st.session_state.std_dev
                    }])
                    
                    # Přidání do Session State
                    st.session_state.calibration_data = pd.concat([st.session_state.calibration_data, new_row], ignore_index=True)
                    st.success(f"Sada '{name_input}' (X={value_x_input} {unit_x_input}) byla uložena do kalibrace.")


# ------------------------------------------------------------------------------------------------
# --- ZÁLOŽKA 2: KALIBRACE (LINEÁRNÍ REGRESE) ---
# ------------------------------------------------------------------------------------------------

with tab2:
    st.header("2. Lineární Regrese (Kalibrace)")
    
    cal_df = st.session_state.calibration_data
    
    # --- SEKCE STAŽENÍ ---
    st.subheader("Správa a výběr kalibračních dat")

    if not cal_df.empty:
        csv_data = cal_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Stáhnout kalibrační sady (CSV)",
            data=csv_data,
            file_name='kalibrace_droplet_analysis.csv',
            mime='text/csv',
            help="Stáhne aktuálně uložené sady do CSV souboru pro trvalé uložení."
        )
    else:
        st.info("Nejprve uložte alespoň jednu sadu na záložce 1, abyste mohli data stáhnout.")
        
    st.markdown("---") 
    
    
    if cal_df.empty:
        st.warning("Žádná kalibrační data nebyla uložena. Vraťte se na záložku 1 a uložte sady.")
    else:
        st.subheader("Uložené sady a výběr pro regresi")
        
        # --- Interaktivní Tabulka pro Výběr Dat (Oprava: Key se provede jen zde) ---
        cal_df_with_select = cal_df.copy()
        cal_df_with_select.insert(0, 'Select', True) 
        
        # Klíč 'cal_editor' je nyní umístěn ve větvi 'else', kde se data editor skutečně zobrazuje.
        edited_df = st.data_editor(
            cal_df_with_select,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Zahrnout do regrese?",
                    default=True,
                    help="Vyberte sady, které chcete použít pro fitování."
                ),
                "Avg_Height_V": st.column_config.NumberColumn("Průměrná Výška (V)", format="%.4f"),
                "STD_Dev_V": st.column_config.NumberColumn("STD (V)", format="%.4f"),
            },
            disabled=['Name', 'Value_X', 'Unit_X', 'Peak_Count', 'Avg_Height_V', 'STD_Dev_V'],
            hide_index=True,
            key="cal_editor" # Zde je umístěn klíč
        )

        # Filtrování vybraných dat
        selected_data = edited_df[edited_df['Select']]
        
        if selected_data.empty:
            st.warning("Pro kalibraci vyberte alespoň jednu sadu dat.")
        elif len(selected_data) < 2:
            st.warning("Pro lineární regresi je potřeba vybrat **alespoň dvě sady dat**.")
        else:
            
            # --- ŘAZENÍ DAT ---
            selected_data = selected_data.sort_values(by='Value_X', ascending=True)
            
            # --- Zajištění jednotek ---
            unique_units = selected_data['Unit_X'].unique()
            if len(unique_units) > 1:
                st.error(f"Nelze provést regresi. Vybrané sady mají různé jednotky X: {', '.join(unique_units)}. Vyberte sady se stejnou jednotkou.")
            else:
                
                # Příprava dat pro regresi a výpočet R^2
                X = selected_data['Value_X'].values
                Y = selected_data['Avg_Height_V'].values
                slope, intercept, r_value, p_value, std_err = linregress(X, Y)
                r_squared = r_value**2
                
                st.subheader("📈 Výsledky Lineární Kalibrace")
                
                col_eq, col_r2 = st.columns(2)
                with col_eq:
                    st.metric(label="Rovnice přímky (Y = aX + b)", value=f"Y = {slope:.4f}X + {intercept:.4f}")
                with col_r2:
                    st.metric(label="Koeficient determinace (R²)", value=f"{r_squared:.4f}")
                    
                # Vykreslení kalibrace
                fig_cal = px.scatter(
                    selected_data, 
                    x='Value_X', 
                    y='Avg_Height_V', 
                    error_y='STD_Dev_V', 
                    title=f"Lineární Kalibrace (Jednotka X: {unique_units[0]})",
                    labels={'Value_X': f"Hodnota X ({unique_units[0]})", 'Avg_Height_V': "Průměrná výška (V)"}
                )
                
                # Přidání fitované přímky
                X_fit = np.linspace(X.min() * 0.9, X.max() * 1.1, 100)
                Y_fit = slope * X_fit + intercept
                
                fig_cal.add_trace(go.Scatter(x=X_fit, y=Y_fit, mode='lines', name=f'Fit (R²={r_squared:.4f})', line=dict(dash='dash')))
                
                st.plotly_chart(fig_cal, use_container_width=True)