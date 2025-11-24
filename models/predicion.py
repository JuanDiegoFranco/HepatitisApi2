def predecir_paciente(model_rl, scaler, nuevo_paciente):
    nuevo_df_scaled = scaler.transform([nuevo_paciente])
    
    prediccion_nuevo = model_rl.predict(nuevo_df_scaled)
    estado = 'Vive' if prediccion_nuevo[0] == 1 else 'Muere'
    
    probabilidades = model_rl.predict_proba(nuevo_df_scaled)
    probabilidad_vive = float(probabilidades[0][0] * 100)
    probabilidad_muere = float(probabilidades[0][1] * 100)

    print(f"La predicción para el nuevo paciente es: {estado}")
    print(f"Probabilidad de que el paciente viva: {probabilidad_vive:.2f}%")
    print(f"Probabilidad de que el paciente muera: {probabilidad_muere:.2f}%")

    return {
        "estado": estado, 
        "probabilidad_vive": probabilidad_vive, 
        "probabilidad_muere": probabilidad_muere,
    }