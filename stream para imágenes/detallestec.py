import streamlit as st
from PIL import Image

st.set_page_config(page_title="Detalles Técnicos", layout="wide")

st.title("🔧 Detalles Técnicos del Proyecto")

st.markdown("""
En esta sección se presentan resultados detallados de los modelos utilizados en los distintos clústeres de viviendas.
""")

# Imagen 1: Comparativa de modelos
st.subheader("📊 Comparativa de Modelos por Clúster")
imagen_resultados = Image.open("Tabla_Marta_1.jpg")
ancho_deseado = 600
alto_redimensionado = int(ancho_deseado * imagen_resultados.height / imagen_resultados.width)
imagen_resultados = imagen_resultados.resize((ancho_deseado, alto_redimensionado))

st.image(imagen_resultados, caption="Tabla 1 - Comparativa modelos", use_container_width=False)
with st.expander("Descripción"):
    st.markdown("Observamos que en dos de los 4 clústeres finales el mejor modelo para entrenar es el RandomForest.")

# Imagen 2: Mejores parámetros
imagen_resultados_2 = Image.open("Tabla_Marta_2 .jpg")
alto_redimensionado_2 = int(ancho_deseado * imagen_resultados_2.height / imagen_resultados_2.width)
imagen_resultados_2 = imagen_resultados_2.resize((ancho_deseado, alto_redimensionado_2))

st.image(imagen_resultados_2, caption="Tabla 2 - Mejores parámetros por clúster", use_container_width=False)
with st.expander("Descripción"):
    st.markdown("""
    Aquí mostramos los mejores parámetros encontrados mediante búsqueda de hiperparámetros, junto con su respectiva puntuación en validación cruzada (mejor_score_cv).  
    Como se puede observar, las métricas no son especialmente altas, lo cual se debe principalmente a la escasez de datos disponibles para el entrenamiento del modelo.  
    El modelo más prometedor alcanzó una puntuación de validación cruzada, lo cual indica cierto poder predictivo, pero también refleja margen de mejora.  
    En algunos casos, observamos una caída considerable en el desempeño, lo que sugiere que ciertos conjuntos de hiperparámetros no generalizan bien con tan pocos datos.  
    Estos resultados refuerzan la importancia de contar con un conjunto de datos más amplio y representativo para obtener modelos más robustos y fiables.
    """)
