import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pickle
from math import pi
import lightgbm
import joblib
import json


def filtrar_propiedades(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = (
        df['superficie construida'].between(f['metros'][0], f['metros'][1]) &
        df['habitaciones'].between(f['habitaciones'][0], f['habitaciones'][1])
    )
    if f['barrios'] and 'Todos' not in f['barrios']:
        mask &= df['barrio'].isin(f['barrios'])
    if f['antiguedad'] and 'antigüedad' in df.columns:
        mask &= df['antigüedad'].isin(f['antiguedad'])
    return df.loc[mask].copy()

def show_main_page():
    col1, _, col_big = st.columns([2, 0.5, 5.5])
    with col1:
        st.image("orangutan.png", caption="¡Selecciona los parámetros en el sidebar! →", width=325)
    with col_big:
        st.title("Predicción de Precios de Viviendas")
        st.markdown(
            "Bienvenido/a. Esta app permite predecir el precio de una propiedad según sus características." \
            "\n\n" \
            "- En el menú desplegable de la izquierda puedes filtrar por tipo de operación (venta o alquiler), metros cuadrados, número de habitaciones, barrios y antigüedad." \
            "\n\n" \
            "- En la pestaña donde se selcciona la sección pordás seleccionar mapa de pisos, métricas, gráficas o predictivo." \
            "\n\n" \
            "- En la sección de Mapa de pisos podrás ver un mapa interactivo con los inmuebles que cumplen con los filtros seleccionados." \
            "\n\n" \
            "- En la sección de métricas podrás ver estadísticas sobre los inmuebles seleccionados, como el número de barrios, el máximo de habitaciones, el número de pisos, el máximo de metros cuadrados y el rating energético más frecuente." \
            "\n\n" \
            "- En la sección de Exploratory Data Analysis podrás ver distintas gráficas para comprender el mercado moviliario actual." \
            "\n\n" \
            "- En la seción predictivo podrás intriducir los datoss de tu inmueble para que nuestro modelo de I.A prediga el precio óptimo de tu viviendS." \
        )
        with st.expander("ℹ️ Acerca del grupo / About Us"):
            st.markdown("""
### 👨‍💻 Integrantes del grupo

- **Akira García** – Junior Data Scientist | Ingeniero Informático | Análisis de Datos & Machine Learning  
  [LinkedIn](https://www.linkedin.com/in/akiragarcialuis/) | [GitHub](https://github.com/akiraglhola)

- **Marta Rivas** – Porque se lo merece  
  [LinkedIn](https://www.linkedin.com/in/MartaRivas) | [GitHub](https://github.com/MartaRivas13)

- **Héctor Frutos** – Es buena gente  
  [LinkedIn](https://www.linkedin.com/in/HectorFrutos) | [GitHub](https://github.com/HFrutos)

- **Jorge Arriaga** – Porque tiene que haber de tó  
  [LinkedIn](https://www.linkedin.com/in/JorgeArriaga) | [GitHub](https://github.com/Jorge-Arriaga)

Proyecto realizado como parte del curso de *Machine Learning aplicado a Datos Inmobiliarios*.

✉️ Para más información, contacta a: pftttttt@example.com
            """, unsafe_allow_html=True)
        with st.expander("🗄️ Arquitectura de la base de datos"):
            st.markdown('<a href="/Arquitectura" target="_self">📄 Ver documentación completa →</a>', unsafe_allow_html=True)

# Cargar niveles categóricos
with open('sale_classifier_features.json', 'r', encoding='utf-8') as f:
    cat_levels = json.load(f)

def aplicar_cat_levels(df: pd.DataFrame) -> pd.DataFrame:
    for col, levels in cat_levels.items():
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.set_categories(levels)
    return df

def main():
    st.set_page_config(page_title='house-price-prediction', page_icon='🏠', layout='wide')

    # Cargar datos
    df_venta = pd.read_csv('madrid_sale_properties_cleaned.csv')
    df_alquiler = pd.read_csv('madrid_rental_properties_cleaned.csv')
    df_venta = aplicar_cat_levels(df_venta)
    df_alquiler = aplicar_cat_levels(df_alquiler)

    # Cargar modelos
    with open('sale_property_classifier.pkl', 'rb') as f:
        model_venta = pickle.load(f)
    with open('rental_property_classifier.pkl', 'rb') as f:
        model_alquiler = pickle.load(f)

#Definir clusteres 
    cluster_info_venta = {
        0: " **Residencial Familiar con Extras** – Propiedades grandes, precio elevado, nuevas, en urbanización con piscina y garaje.",
        1: " **Piso Básico y Económico** – El más barato, en zonas menos céntricas, pequeño, antiguo, con pocos extras y sin amueblar.",
        2: " **Apartamento Céntrico Amueblado** – Tamaño mediano o pequeño, completamente amueblado, en barrios con alquiler alto, con ascensor.",
        3: " **Vivienda de Lujo Exclusivo** – El más caro y grande, ubicación y calidades de lujo, con todos los extras.",
        4: "**Apartamento Premium (Ubicación Prime)** – Precio elevado, ubicación prime, excelente estado (reformado o nuevo), exterior y con ascensor."
    }
    cluster_info_renta = {
        0: " **Piso Señorial Clásico** – Grande, precio elevado, finca antigua, con ascensor, pero sin piscina/garaje.",
        1: " **Apartamento Estándar** – El más económico, tamaño reducido y menos extras. El apartamento 'típico' de Madrid.",
        2: " **Propiedad Singular** – Anomalía. Extremadamente grande y caro. ",
        3: " **Lujo Moderno (Full Equip)** – El más caro, grande, obra nueva o reciente, y equipado con piscina, garaje y terraza.",
        4: "**Premium Reformado** – Precio elevado, ubicación prime, finca antigua pero en excelente estado (reformado), y amueblado."
    }

  
    # Convertir columnas a categóricas
    for df in [df_venta, df_alquiler]:
        df['barrio'] = df['barrio'].astype('category')
        if 'antigüedad' in df.columns:
            df['antigüedad'] = df['antigüedad'].astype('category')

    # Inicializar estados de sesión si no existen
    if 'temp_filters' not in st.session_state:
        st.session_state.temp_filters = dict(section='Página principal', tipo_operacion='venta', metros=(50,150), habitaciones=(1,4), barrios=['Todos'], antiguedad=[])
    if 'applied_filters' not in st.session_state:
        st.session_state.applied_filters = st.session_state.temp_filters.copy()
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'main_page'

    with st.sidebar.form("filtros"):
        tf = st.session_state.temp_filters
        tf['tipo_operacion'] = st.selectbox('Tipo de operación', ['venta', 'alquiler'], index=['venta', 'alquiler'].index(tf['tipo_operacion']))
        tf['metros'] = st.slider('Metros cuadrados', 20, 600, tf['metros'])
        tf['habitaciones'] = st.slider('Habitaciones', 1, 8, tf['habitaciones'])
        df_temp = df_venta if tf['tipo_operacion']=='venta' else df_alquiler
        barrios = ['Todos'] + sorted(df_temp['barrio'].cat.categories)
        tf['barrios'] = st.multiselect('Barrios', barrios, default=tf['barrios'])
        if 'antigüedad' in df_temp.columns:
            antig = df_temp['antigüedad'].cat.categories.tolist()
            tf['antiguedad'] = st.multiselect('Antigüedad', antig, default=tf['antiguedad'])
        else:
            tf['antiguedad'] = []
        tf['section'] = st.selectbox(
            'Selecciona la sección',
            ['Página principal', 'Comparativa de pisos', 'Exploratory Data Analysis', 'Predictivo'],
            index=['Página principal', 'Comparativa de pisos', 'Exploratory Data Analysis', 'Predictivo'].index(tf['section'])
        )

        submitted = st.form_submit_button("Aplicar filtros")

    if submitted:
        st.session_state.applied_filters = st.session_state.temp_filters.copy()
        st.session_state.current_view = 'main_page' if tf['section']=='Página principal' else 'section_content'

    if st.session_state.current_view == 'main_page':
        show_main_page()

    f = st.session_state.applied_filters
    df_sel = df_venta if f['tipo_operacion']=='venta' else df_alquiler

    need_recalc = 'df_filtrado' not in st.session_state or st.session_state.df_filtrado.filters != f
    if need_recalc:
        filtered_df = filtrar_propiedades(df_sel, f)
        filtered_df.filters = f.copy()
        st.session_state.df_filtrado = filtered_df

    filtro = st.session_state.df_filtrado

    if filtro.empty:
        st.warning('No hay resultados para los filtros seleccionados.')

# comparativa de prisos

    if f['section'] == 'Comparativa de pisos':
        st.title('Mapa de pisos en Madrid')

        tipo_operacion = st.radio('Tipo de operación', ['Venta', 'Alquiler'], horizontal=True)

        st.subheader('Pisos que coinciden con tu búsqueda')
        st.markdown(f'**{len(filtro)} pisos encontrados**')

        # Lista de pisos para comparar
        st.subheader("Selecciona pisos para comparar")

        filtro['Resumen'] = filtro.apply(lambda x: f"{x['price_eur']:,}€ - {x['habitaciones']} hab - {x['superficie construida']} m² - {x['barrio']}", axis=1)
        seleccion = st.multiselect("Elige pisos", options=filtro.index, format_func=lambda idx: filtro.loc[idx, 'Resumen'])

        if filtro.empty:
            st.warning('No hay resultados para los filtros seleccionados.')
            st.stop()

        if seleccion:
            st.success(f"{len(seleccion)} piso(s) seleccionados")
        else:
            st.info("No has seleccionado ningún piso manualmente.")

        # Limpieza de coordenadas
        filtro = filtro.dropna(subset=['lat', 'lon'])
        filtro = filtro[(filtro['lat'].between(-90, 90)) & (filtro['lon'].between(-180, 180))]

        # Mapa
        m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)

        if not filtro.empty:
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in filtro.iterrows():
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=f"""
                        <b>Precio:</b> {row['price_eur']:,} €<br>
                        <b>Habitaciones:</b> {row['habitaciones']}<br>
                        <b>m²:</b> {row['superficie construida']}<br>
                        <b>Barrio:</b> {row['barrio']}
                    """,
                    icon=folium.Icon(color='blue', icon='home')
                ).add_to(marker_cluster)

        folium_static(m, width=800, height=600)

        # Comparativa Radar
        st.subheader("Comparativa de características (Radar)")

        variables = ['price_eur', 'habitaciones', 'superficie construida']
        labels = ['Precio (€)', 'Habitaciones', 'Superficie (m²)']

        # Selección para radar (con al menos 3 pisos)
        if seleccion:
            comparacion_df = filtro.loc[seleccion]
        elif len(filtro) >= 3:
            comparacion_df = filtro.sample(3, random_state=42)
            st.info("Se han seleccionado 3 pisos al azar para la comparativa.")
        elif not filtro.empty:
            comparacion_df = filtro
            st.info("Menos de 3 pisos disponibles. Se compararán todos los que hay.")
        else:
            st.warning("No hay pisos disponibles para comparar.")
            st.stop()

        # Normalización
        scaler = MinMaxScaler()
        valores_normalizados = scaler.fit_transform(comparacion_df[variables])

        # Radar plot
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        angles = [n / float(len(variables)) * 2 * pi for n in range(len(variables))]
        angles += angles[:1]

        for i, row in comparacion_df.iterrows():
            values = valores_normalizados[comparacion_df.index.get_loc(i)].tolist()
            values += values[:1]
            ax.plot(angles, values, label=row['Resumen'])
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)

# Eda
    if f['section']=='Exploratory Data Analysis':
        st.title("Exploratory Data Analisis de los Pisos")
        if filtro.empty:
            st.warning("No hay datos...")
        else:
            st.metric('Número de barrios', filtro['barrio'].nunique())
            st.metric('Máx habitaciones', filtro['habitaciones'].max())
            st.metric('Número de pisos', len(filtro))
            st.metric('Máx metros (m²)', f"{filtro['superficie construida'].max():.1f}")
            if 'energy_consumption_rating' in filtro.columns and filtro['energy_consumption_rating'].notna().any():
                st.metric('Rating energético más frecuente', filtro['energy_consumption_rating'].mode()[0])
            medias = filtro.groupby('barrio')[['price_eur','superficie construida']].mean()
            st.subheader('Precio y superficie media por barrio')
            st.dataframe(medias.rename(columns={'price_eur':'Precio medio (€)','superficie construida':'Superficie media (m²)'}))
            fig = px.bar(medias.sort_values('price_eur').reset_index(), x='barrio', y='price_eur', title='Precio medio por barrio')
            st.plotly_chart(fig)
#Predictivo 
    if f['section'] == 'Predictivo':
        st.title("Predicción de precio de propiedades")

        tipo_operacion = st.radio("Selecciona el tipo de operación", ['Venta', 'Alquiler'])
        if tipo_operacion == 'Venta':
            df_clustered = pd.read_csv('sale_properties_clustered.csv')
        else:
            df_clustered = pd.read_csv('rental_properties_clustered.csv')

            # Asegurarnos que todos los valores son strings (evita comparar float y str)
        df_clustered['distrito'] = df_clustered['distrito'].fillna('Desconocido').astype(str)

        # Selector de distrito
        distritos = sorted(df_clustered['distrito'].unique())
        distrito_seleccionado = st.selectbox("Selecciona un distrito", distritos)
        df_clustered['distrito'] = df_clustered['distrito'].astype(str)


        # Inputs numéricos
        superficie = st.slider("Superficie construida (m²)", 20, 300, 80)
        habitaciones = st.number_input("Número de habitaciones", 1, 10, 3)
        banos = st.number_input("Número de baños", 1, 5, 2)
        antiguedad = st.slider("Antigüedad del inmueble (años)", 0, 100, 20)

        # Inputs booleanos
        adaptado = st.checkbox("Adaptado movilidad reducida")
        aire = st.checkbox("Aire acondicionado")
        armarios = st.checkbox("Armarios empotrados")
        ascensor = st.checkbox("Ascensor")
        balcon = st.checkbox("Balcón")
        calefaccion = st.checkbox("Calefacción")
        chimenea = st.checkbox("Chimenea")
        cocina_eq = st.checkbox("Cocina equipada")
        exterior = st.checkbox("Exterior")
        garaje = st.checkbox("Garaje")
        jardin = st.checkbox("Jardín")
        piscina = st.checkbox("Piscina")
        puerta = st.checkbox("Puerta blindada")
        seguridad = st.checkbox("Sistema de seguridad")
        terraza = st.checkbox("Terraza")
        trastero = st.checkbox("Trastero")
        vidrios = st.checkbox("Vidrios dobles")
        planta = st.number_input("Número de planta", 0, 20, 0)
        conservacion = st.selectbox("Conservación", [0, 1, 2])

        # Mobiliario
        amueblado_op = st.selectbox("¿Está amueblado?", ['False', 'True', 'Nan'])
        amueblado_False = int(amueblado_op == 'False')
        amueblado_True = int(amueblado_op == 'True')
        amueblado_nan = int(amueblado_op == 'Nan')

        # Botón para realizar la predicción
        if st.button("Predecir"):
            input_dict = {
                'distrito': distrito_seleccionado,
                'superficie_construida': superficie,
                'habitaciones': habitaciones,
                'banos': banos,
                'antiguedad': antiguedad,
                'adaptado_movilidad_reducida': int(adaptado),
                'aire_acondicionado': int(aire),
                'armarios_empotrados': int(armarios),
                'ascensor': int(ascensor),
                'balcon': int(balcon),
                'calefaccion': int(calefaccion),
                'chimenea': int(chimenea),
                'cocina_equipada': int(cocina_eq),
                'exterior': int(exterior),
                'garaje': int(garaje),
                'jardin': int(jardin),
                'piscina': int(piscina),
                'puerta_blindada': int(puerta),
                'sistema_seguridad': int(seguridad),
                'terraza': int(terraza),
                'trastero': int(trastero),
                'vidrios_dobles': int(vidrios),
                'planta_numerica': planta,
                'conservacion': conservacion,
                'amueblado_False': amueblado_False,
                'amueblado_True': amueblado_True,
                'amueblado_nan': amueblado_nan,
            }

            input_data = pd.DataFrame([input_dict])

            # Realizar la predicción
            prediccion = model.predict(input_data)

            # Mostrar el resultado
            st.write(f"💰 Precio estimado: **{prediccion[0]:,.2f} €**")


if __name__ == '__main__':
    main()