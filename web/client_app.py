import streamlit as st
import pandas as pd
import requests
import plotly.express as px # Nueva librería para gráficos
import io
import os

st.set_page_config(page_title="MLOps", layout="wide")

st.title("MLops ")
st.sidebar.header("Configuración")
st.sidebar.info("Este portal se conecta a un microservicio de IA desplegado en Docker.")

uploaded_file = st.file_uploader(type="csv")

if uploaded_file:
    df_original = pd.read_csv(uploaded_file)
    
    if st.button("Magia!"):
        api_candidates = []
        env_api = os.getenv("URL_API")
        if env_api:
            api_candidates.append(env_api)
        api_candidates.extend([
            "http://api:8000/predict-csv",
            "http://localhost:8000/predict-csv",
        ])
        
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = None
            last_error = None
            for api_url in api_candidates:
                try:
                    response = requests.post(api_url, files=files, timeout=30)
                    break
                except requests.RequestException as request_error:
                    last_error = request_error
                    continue

            if response is None:
                raise RuntimeError(f"No se pudo conectar con la API: {last_error}")
            
            if response.status_code == 200:
                result_data = response.json()
                df_result = pd.DataFrame(result_data)

                if "prediction" not in df_result.columns:
                    st.error("La respuesta no contiene la columna 'prediction'.")
                    st.stop()

                st.subheader("Resultados de predicción")
                st.dataframe(df_result, use_container_width=True)

                st.subheader("Distribución de predicciones")
                fig_dist = px.histogram(
                    df_result,
                    x="prediction",
                    nbins=30,
                    title="Distribución de predicciones de valor de mercado",
                )
                fig_dist.update_layout(bargap=0.05, xaxis_title="Predicción", yaxis_title="Frecuencia")
                st.plotly_chart(fig_dist, use_container_width=True)

                st.subheader("Relación técnica y valor predicho")
                if all(col in df_result.columns for col in ["overall", "potential"]):
                    fig_scatter = px.scatter(
                        df_result,
                        x="overall",
                        y="potential",
                        color="prediction",
                        title="Overall vs Potential coloreado por predicción",
                        labels={"overall": "Overall", "potential": "Potential", "prediction": "Predicción"},
                    )
                elif "overall" in df_result.columns:
                    fig_scatter = px.scatter(
                        df_result,
                        x="overall",
                        y="prediction",
                        color="prediction",
                        title="Overall vs Predicción",
                        labels={"overall": "Overall", "prediction": "Predicción"},
                    )
                else:
                    fig_scatter = px.scatter(
                        df_result,
                        x=df_result.index,
                        y="prediction",
                        color="prediction",
                        title="Índice vs Predicción",
                        labels={"x": "Índice", "prediction": "Predicción"},
                    )
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.subheader("Resumen estadístico de predicciones")
                st.dataframe(df_result["prediction"].describe().to_frame(name="prediction"))

                csv_buffer = io.StringIO()
                df_result.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Descargar resultados como CSV",
                    data=csv_buffer.getvalue(),
                    file_name="predicciones_resultado.csv",
                    mime="text/csv",
                )
                
            else:
                st.error("Error en el servidor de IA. Revisa el formato del CSV.")
        except Exception as e:
            st.error(f"Error de conexión: {e}")