# Streamlit UI to submit CSVs for async inference and download predictions
import streamlit as st
import requests
import time
import pandas as pd

API_BASE_URL = "http://inference:8000" # FastAPI service base URL

st.title("ML Inference")

task_choice = st.selectbox("Choose Task", ["Application (app)", "Attribution (att)"])
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file and st.button("Run"):
    # Read the original dataframe
    df_raw = pd.read_csv(uploaded_file)
    
    # Send the uploaded CSV to the API
    uploaded_file.seek(0)
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    task_key = "app" if "Application" in task_choice else "att"
    
    try:
        # 1. Send request to FastAPI
        response = requests.post(f"{API_BASE_URL}/predict/{task_key}", files=files)
        response.raise_for_status()
        task_id = response.json().get("task_id")
        
        # 2. Poll for results in the background
        with st.spinner("Processing traffic data..."):
            while True:
                res_node = requests.get(f"{API_BASE_URL}/result/{task_id}").json()
                
                if res_node["status"] == "SUCCESS":
                    raw_results = res_node["results"]
                    
                    # Normalize results into a list
                    if isinstance(raw_results, dict):
                        sorted_keys = sorted(raw_results.keys(), key=int)
                        predictions = [raw_results[k] for k in sorted_keys]
                    else:
                        predictions = raw_results
                    
                    st.success("Analysis Complete!")
                    st.write("Preview of first 10 predictions:")
                    st.write(predictions[:10])
                    
                    # 3. Final submission file preparation (attach predictions + download CSV)
                    if len(predictions) == len(df_raw):
                        df_raw['prediction'] = predictions
                        csv_data = df_raw.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Submission CSV",
                            data=csv_data,
                            file_name=f"radcom_{task_key}_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Error: Prediction count does not match input rows.")
                    break
                
                elif res_node["status"] == "FAILURE":
                    st.error("Worker process failed. Check server logs.")
                    break
                    
                time.sleep(1) # Wait 1 second before next poll (poll interval)
                
    except Exception as e:
        st.error(f"System Error: {e}")