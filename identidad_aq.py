import streamlit as st

def identidad_AQ():
    import streamlit as st
    
    LOGO_PATH = "AlmaQuant.png"

    # --- Logo Centrado y Título ---
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(LOGO_PATH, width=300)

        st.title('¿Quiénes somos?')

    st.markdown("---")

    # --- 1. Nuestra Misión ---
    st.header("1. Nuestra Misión: Transformar Datos en Decisiones")
    st.write(
        """
        En **Alma Quant**, creemos firmemente que las mejores decisiones financieras surgen de un 
        entendimiento profundo de los datos y del comportamiento de los mercados. Nuestra misión es 
        **convertir la complejidad financiera en estrategias claras, cuantificables y orientadas a resultados**.

        Buscamos que cada inversor, sin importar su nivel de experiencia, acceda a herramientas 
        profesionales basadas en **matemáticas, estadística y ciencia de datos**, para gestionar su capital 
        con confianza y precisión.
        """
    )

    # --- 2. Metodología y Filosofía Cuantitativa ---
    st.header("2. Nuestra Metodología Cuantitativa")
    st.write(
        """
        Todo nuestro análisis se sustenta en un enfoque cuantitativo riguroso. Implementamos modelos 
        avanzados de **Optimización de Portafolios**, como:
        - La **Teoría Moderna de Portafolios** (Markowitz)
        - Maximización del **ratio de Sharpe**
        - Portafolios de **mínima varianza**

        Nuestro principio es sencillo: **no especulamos… modelamos**.  
        Cada decisión se analiza mediante métricas objetivas y criterios estadísticos.
        """
    )

    # --- 3. Equipo ---
    st.header("3. El Equipo Detrás del Análisis")
    st.markdown(
        """
        Alma Quant está conformado por un equipo multidisciplinario que combina matemáticas aplicadas, 
        estadística y ciencias actuariales. Esta mezcla de perfiles nos permite analizar los mercados 
        desde una perspectiva técnica, rigurosa y coherente con los estándares modernos de gestión de inversiones.
        """
    )

    st.subheader("Fundadores")
    col_emiliano, col_eduardo, col_monserrat = st.columns(3)

    with col_emiliano:
        st.markdown("### **Emiliano**")
        st.markdown("*Matemático Aplicado (M.A.)*")
        st.info(
            "Especialista en **Modelado Cuantitativo y Optimización**. "
            "Lidera el diseño e implementación de los modelos que construyen la frontera eficiente y las métricas clave."
        )

    with col_eduardo:
        st.markdown("### **Eduardo**")
        st.markdown("*Matemático Aplicado (M.A.)*")
        st.info(
            "**Experto en Estadística y Validación de Modelos**. "
            "Se enfoca en la robustez de los datos, backtesting y análisis comparativo de estrategias."
        )

    with col_monserrat:
        st.markdown("### **Montserrat**")
        st.markdown("*Actuaria (Act.)*")
        st.info(
            "Especialista en **Gestión de Riesgos y Análisis Estocástico**. "
            "Aporta el enfoque crítico para evaluar volatilidad, riesgo extremo y sensibilidad de portafolios."
        )

    st.markdown("---")

    # --- CTA ---
    st.subheader("Explora Nuestro Trabajo Cuantitativo")
    st.info(
        "Puedes visitar la sección **Nuestro punto de vista** para conocer nuestro análisis del mercado, "
        "o entrar a **Crea tu propio portafolio** para interactuar con los modelos de optimización."
    )