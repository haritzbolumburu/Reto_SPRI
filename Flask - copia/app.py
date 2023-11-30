from flask import Flask, render_template, request, session, make_response
import bbdd.sql_prueba as sql

import joblib
from openpyxl import Workbook
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

import os
import numpy as np
import io

df = pd.read_csv('../Datos/Limpios/FCF.csv')
df_clasif = pd.read_csv('../Datos/Limpios/df_clasif.csv')
df_regr = pd.read_csv('../Datos/Limpios/df_regr.csv')

kpis20 = pd.read_csv('../Datos/Limpios/kpis20.csv')
kpis21 = pd.read_csv('../Datos/Limpios/kpis21.csv')

carp_modelos_adqui = '../Modelos_FLASK/adquisicion'
carp_modelos_val = '../Modelos_FLASK/valoracion'

modelos_adqui = os.listdir(carp_modelos_adqui)
modelos_val = os.listdir(carp_modelos_val)

COLUMNAS_REGRESION = ['ratio_periodo_prom_cobro', 'last_funding',
       'Período de cobro (días) días', 'Deudas financieras mil EUR',
       'Endeudamiento (%) %', 'Número empleados',
       'Costes de los trabajadores / Ingresos de explotación (%) %',
       'Acreedores comerciales mil EUR', 'dias_operando',
       'Pasivo fijo mil EUR', 'Rotación de las existencias %', 'total_rounds',
       'ratio_deuda_activos', 'Existencias mil EUR', 'Capital social mil EUR',
       'total_funding', 'Inmovilizado mil EUR', 'Total activo mil EUR',
       'Total pasivo y capital propio mil EUR',
       'Ratios de autonomía financiera a medio y largo plazo %',
       'Tesorería mil EUR', 'Activo circulante mil EUR']

COLUMNAS_ADQUISICION = [ 'dias_operando',
       'Ratios de autonomía financiera a medio y largo plazo %',
       'EBITDA mil EUR', 'Acreedores comerciales mil EUR',
       'ratio_ebitda_activos', 'ratio_ebitda_patrimonio',
       'Rentabilidad sobre el activo total (%) %',
       'Rentabilidad económica (%) %', 'Resultado financiero mil EUR',
       'ratio_prueba_acida', 'Dotaciones para amortiz. de inmovil. mil EUR',
       'ratio_ventas_ebitda', 'ratio_deuda_patrimonio', 'Endeudamiento (%) %',
       'ratio_deuda_activos', 'Deudas financieras mil EUR',
       'Costes de los trabajadores / Ingresos de explotación (%) %',
       'Rotación de activos netos %', 'Inmovilizado mil EUR',
       'Valor agregado mil EUR', 'Ratio de liquidez %', 'Ratio de solvencia %',
       'Liquidez general %']

app = Flask(__name__, template_folder="templates")

app.secret_key = 'jsndk'

# Crear base de datos y tabla de alumnos
sql.crear_bbdd_tabla()

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/servicios')
def servicios():
    return render_template('servicios.html')


@app.route("/formulario", methods = ["GET", "POST"])
def formulario():
    if request.method == "POST":
        session['selected_empresa'] = request.form.get('Codigo_NIF')
        session['selected_loc'] = request.form.get('Localidad')
        
        if (df[(df['Codigo_NIF'] == session['selected_empresa']) & (df['Localidad'] == session['selected_loc'])].all(axis=None) == True):
            return render_template("formulario_mal.html") 
         
        sql.insertar_empresa(session['selected_empresa'], session['selected_loc'])
        return render_template('servicios_empresas.html')
    else:
        return render_template("formulario.html")


@app.route('/servicios-empresas')
def servicios_empresas():
    return render_template('servicios_empresas.html')

@app.route('/info-empresas')
def info_empresas():
    empresa = df[(df["Localidad"] == session['selected_loc']) & (df["Codigo_NIF"] == session['selected_empresa']) & (df['year'] == 2021)]
    empresa = empresa[['Codigo_NIF', 'Nombre_sabi', 'Localidad', 'Fecha constitucion', 'Codigo primario CNAE 2009',
       'Forma juridica detallada', 'Estado detallado']]

    return render_template('info_empresa.html', 
                                            nombre_empresa = empresa[empresa['Codigo_NIF']==session['selected_empresa']]['Nombre_sabi'].iloc[0],
                                            column_names = empresa.columns.values, 
                                            row_data = list(empresa.values.tolist()), 
                                            zip = zip)
        

@app.route('/valoracion', methods = ['GET', 'POST'])
def valoracion():
    if request.method == "POST":
        data = df[df['Codigo_NIF'] == session['selected_empresa']][COLUMNAS_REGRESION]
        data = data.head(1).values[0]
        data = np.log(data + 1)
        data = np.nan_to_num(data)
        data = list(data)

        modelo = request.form.get('modelo_val')

        if modelo == 'modelo_regresion.sav':
            model = joblib.load('../Modelos_FLASK/valoracion/modelo_regresion.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            resultado = np.exp(resultado)
            resultado = resultado[0]
            return render_template('predic_val.html', resultado = resultado)
        
        else:
            model = joblib.load('../Modelos_FLASK/valoracion/modelo_regresion2.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            resultado = np.exp(resultado)
            resultado = resultado[0]
            return render_template('predic_val.html', resultado = resultado)
  

    else:
        return render_template('valoracion.html', modelos_val = modelos_val)

@app.route('/adquisicion', methods = ['GET', 'POST'])
def adquisicion():
    if request.method == "POST":
        data = df[df['Codigo_NIF'] == session['selected_empresa']][COLUMNAS_ADQUISICION]
        data = data.head(1).values[0]

        modelo = request.form.get('modelo_adqui')

        if modelo == 'modelo_clasificacion.sav':
            model = joblib.load('../Modelos_FLASK/adquisicion/modelo_clasificacion.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            resultado = resultado[0]
            if resultado == 0:
                resultado = 'No adquirida'
                return render_template('predic_adqui.html', resultado = resultado)
            elif resultado == 1:
                resultado = 'Adquirida parcialmente'
                return render_template('predic_adqui.html', resultado = resultado)
            else:
                resultado = 'Adquirida totalmente'
                return render_template('predic_adqui.html', resultado = resultado)
        
        else:
            model = joblib.load('../Modelos_FLASK/adquisicion/modelo_clasificacion2.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            resultado = resultado[0]
            if resultado == 0:
                resultado = 'No adquirida'
                return render_template('predic_adqui.html', resultado = resultado)
            elif resultado == 1:
                resultado = 'Adquirida parcialmente'
                return render_template('predic_adqui.html', resultado = resultado)
            else:
                resultado = 'Adquirida totalmente'
                return render_template('predic_adqui.html', resultado = resultado)
                    
    else:
        return render_template('adquisicion.html', modelos_adqui = modelos_adqui)

@app.route('/nueva_empresa')
def form_nueva_predic():
    return render_template('servicios_nueva_empresa.html')

@app.route('/datos_adquisicion')
def datos_adqui():
    return render_template('form_adqui.html')

@app.route('/datos_valoracion')
def datos_val():
    return render_template('form_val.html')

@app.route('/descargar_excel_adqui')
def descargar_excel_adqui():
    data = [COLUMNAS_ADQUISICION]

    wb = Workbook()
    ws = wb.active
    for row in data:
        ws.append(row)

    excel_data = io.BytesIO()
    wb.save(excel_data)
    excel_data.seek(0)

    response = make_response(excel_data.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=plantilla_datos_adqui.xlsx'
    response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return response

@app.route('/descargar_excel_val')
def descargar_excel_val():
    data = [COLUMNAS_REGRESION]

    wb = Workbook()
    ws = wb.active
    for row in data:
        ws.append(row)

    excel_data = io.BytesIO()
    wb.save(excel_data)
    excel_data.seek(0)

    response = make_response(excel_data.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=plantilla_datos_val.xlsx'
    response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return response

@app.route('/subir_excel-val', methods=['GET', 'POST'])
def upload_file_val():
    if request.method == 'POST':
        file = request.files['file']

        if file and file.filename.endswith('.xlsx'):

            df_usu = pd.read_excel(file)
            json_data = df_usu.to_dict(orient='records')
            session['df_usu'] = json_data
            return render_template('valoracion_nueva.html', modelos_val = modelos_val)

    return render_template('form_val.html')

@app.route('/subir_excel-adqui', methods=['GET', 'POST'])
def upload_file_adqui():
    if request.method == 'POST':
        file = request.files['file']

        if file and file.filename.endswith('.xlsx'):

            df_usu = pd.read_excel(file)
            json_data = df_usu.to_dict(orient='records')
            session['df_usu'] = json_data
            return render_template('adquisicion_nueva.html', modelos_adqui = modelos_adqui)

    return render_template('form_adqui.html')

@app.route('/valoracion_excel', methods = ['GET', 'POST'])
def valoracion_nueva():
    if request.method == "POST":
        data = pd.DataFrame(session['df_usu'])
        data = np.log(data+1)
        data = np.nan_to_num(data)
        data = list(data)
        
        modelo = request.form.get('modelo_val')
        
        if modelo == 'modelo_regresion.sav':
            model = joblib.load('../Modelos_FLASK/valoracion/modelo_regresion.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            resultado = np.exp(resultado)
            resultado = resultado[0]
            return render_template('predic_val.html', resultado = resultado)
        
        else:
            model = joblib.load('../Modelos_FLASK/valoracion/modelo_regresion2.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            resultado = np.exp(resultado)
            resultado = resultado[0]
            return render_template('predic_val.html', resultado = resultado)
        
    else:
        return render_template('valoracion_nueva.html', modelos_val = modelos_val)

@app.route('/adquisicion_excel', methods = ['GET', 'POST'])
def adquisicion_nueva():
    if request.method == "POST":
        data = pd.DataFrame(session['df_usu'])
        data = data.head(1).values[0]

        modelo = request.form.get('modelo_adqui')

        if modelo == 'modelo_clasificacion.sav':
            model = joblib.load('../Modelos_FLASK/adquisicion/modelo_clasificacion.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            resultado = resultado[0]
            if resultado == 0:
                resultado = 'No adquirida'
                return render_template('predic_adqui.html', resultado = resultado)
            elif resultado == 1:
                resultado = 'Adquirida parcialmente'
                return render_template('predic_adqui.html', resultado = resultado)
            else:
                resultado = 'Adquirida totalmente'
                return render_template('predic_adqui.html', resultado = resultado)
        
        else: 
            model = joblib.load('../Modelos_FLASK/adquisicion/modelo_clasificacion2.sav')
            resultado = model.predict(np.array(data).reshape(1, -1))
            if resultado == 0:
                resultado = 'No adquirida'
                return render_template('predic_adqui.html', resultado = resultado)
            elif resultado == 1:
                resultado = 'Adquirida parcialmente'
                return render_template('predic_adqui.html', resultado = resultado)
            else:
                resultado = 'Adquirida totalmente'
                return render_template('predic_adqui.html', resultado = resultado)

    else:
        return render_template('adquisicion_nueva.html', modelos_adqui = modelos_adqui)


@app.route("/balance")
def graf_balance():
  
    empresa_graf = df[(df["Localidad"] == session['selected_loc']) & (df["Codigo_NIF"] == session['selected_empresa'])]
    empresa_graf_20 = empresa_graf[empresa_graf['year'] == 2020]
    empresa_graf_21 = empresa_graf[empresa_graf['year'] == 2021]
    nombre_empresa = empresa_graf_20[empresa_graf_20['Codigo_NIF']==session['selected_empresa']]['Nombre_sabi'].iloc[0]
    session['nombre_empresa'] = nombre_empresa
    x1 = ['Total Activo']
    x2 = ['Total Pasivo']

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Bar(
        name = 'Activo corriente',
        x = x1,
        y = empresa_graf_20['Activo circulante mil EUR']
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name = 'Activo no corriente',
        x = x1,
        y = empresa_graf_20['Inmovilizado mil EUR']
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name = 'Pasivo corriente',
        x = x2,
        y = empresa_graf_20['Pasivo líquido mil EUR']
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name = 'Pasivo no corriente',
        x = x2,
        y = empresa_graf_20['Pasivo fijo mil EUR']
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name = 'Patrimonio Neto',
        x = x2,
        y = empresa_graf_20['Fondos propios mil EUR']
    ), row=1, col=2)

    #---------------------------------------

    fig.add_trace(go.Bar(
        name = 'Activo corriente',
        x = x1,
        y = empresa_graf_21['Activo circulante mil EUR']
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name = 'Activo no corriente',
        x = x1,
        y = empresa_graf_21['Inmovilizado mil EUR']
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name = 'Pasivo corriente',
        x = x2,
        y = empresa_graf_21['Pasivo líquido mil EUR']
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name = 'Pasivo no corriente',
        x = x2,
        y = empresa_graf_21['Pasivo fijo mil EUR']
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name = 'Patrimonio Neto',
        x = x2,
        y = empresa_graf_21['Fondos propios mil EUR']
    ), row=1, col=2)

    fig.update_layout(updatemenus=[
        dict(
            type="buttons",
            direction="left",
            showactive=True,
            buttons=list([
                dict(label="2020",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, False, False, False, False, False]}]),

                dict(label="2021",
                        method="update",
                        args=[{"visible": [False, False, False, False, False, True, True, True, True, True]}])
            ])
        )
    ])
    fig.update_traces(visible=True if "Button 1" in fig.layout.updatemenus[0].buttons[0]['args'][0]['visible'] else False)
    fig.update_layout(
        title=f'Balance de la empresa {session["nombre_empresa"]}'
    )

    fig.update_layout(barmode='stack')
        
    return render_template('grafico.html', plot=fig.to_html())


@app.route("/kpis")
def graf_kpis():

    data20 = kpis20[kpis20['Codigo_NIF'] == session['selected_empresa']]
    data21 = kpis21[kpis21['Codigo_NIF'] ==  session['selected_empresa']]
    
    data20 = data20[['Rentabilidad', 'Liquidez_Solvencia', 'Eficiencia_Apalancamiento']]
    data21 = data21[['Rentabilidad', 'Liquidez_Solvencia', 'Eficiencia_Apalancamiento']]
    
    data20 = list(data20.head(1).values[0])
    data21 = list(data21.head(1).values[0])

    theta = ['Rentabilidad', 'Liquidez_Solvencia', 'Eficiencia_Apalancamiento']
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
    fig.add_trace(go.Scatterpolar(r=data20, theta=theta, fill='toself', name='2020'), row=1, col=1)
    fig.add_trace(go.Scatterpolar(r=data21, theta=theta, fill='toself', name='2021'), row=1, col=1)

    fig.update_layout(
        title=f'KPIs de la empresa {session["nombre_empresa"]}',
        
    )
    return render_template('kpis.html', plot=fig.to_html())

@app.route("/resultados")
def resultados():
    resultados = sql.consultar_todos()
    print(resultados)
    return render_template("resultados.html", resultados = resultados)


if __name__ == "__main__":
    app.run(debug = True)
