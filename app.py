import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
import psycopg2
from psycopg2 import pool

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '1234')

# Neon connection pool (update with your Neon credentials)
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,
    host=os.environ.get('DB_HOST', 'your-neon-host.neon.tech'),  # From Neon dashboard
    port=os.environ.get('DB_PORT', 5432),
    database=os.environ.get('DB_NAME', 'your_database'),
    user=os.environ.get('DB_USER', 'your_username'),
    password=os.environ.get('DB_PASSWORD', 'your_password'),
    sslmode='require'
)

CSV_URL = "https://drive.google.com/uc?id=1EV4AoEymcBA3FEFgce2-Dq9cBSXu4rIu"
df = pd.read_csv(CSV_URL)
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype(int)

@app.route('/task', methods=['GET', 'POST'])
def task():
    if 'condition' not in session:
        return redirect(url_for('index'))
    
    task_index = session['task_index']
    if task_index >= len(session['tasks']):
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                for response in session['responses']:
                    task_data = next(t for t in session['tasks'] if t['ID'] == response['ID'])
                    revolving_util = float(str(task_data['RevolvingUtilizationOfUnsecuredLines']).replace('%', '')) / 100 if '%' in str(task_data['RevolvingUtilizationOfUnsecuredLines']) else float(task_data['RevolvingUtilizationOfUnsecuredLines'])
                    late_payments = int(task_data['NumberOfTime30-59DaysPastDueNotWorse'])
                    debt_ratio = float(str(task_data['DebtRatio']).replace('%', '')) / 100 if '%' in str(task_data['DebtRatio']) else float(task_data['DebtRatio'])
                    monthly_income = float(str(task_data['MonthlyIncome']).replace('$', '').replace(',', '')) if any(c in str(task_data['MonthlyIncome']) for c in ['$', ',']) else float(task_data['MonthlyIncome'])
                    cur.execute(
                        "INSERT INTO responses (participant_id, condition, initial_decision, final_decision, predicted, confidence_score, level, "
                        "revolving_utilization, late_payments_30_59, debt_ratio, monthly_income) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (response['ID'], response['Condition'], response['Initial_Decision'], 
                         response['Final_Decision'], response['Predicted'], response['Confidence_Score'], 
                         response['Level'], revolving_util, late_payments, debt_ratio, monthly_income)
                    )
                conn.commit()
                print("Data committed to Neon")
        except Exception as e:
            print(f"Database error: {e}")
            conn.rollback()
        finally:
            db_pool.putconn(conn)
        return render_template('end.html')
    
    task_data = session['tasks'][task_index]
    condition = session['condition']
    
    if request.method == 'POST':
        initial_decision = request.form['initial_decision']
        final_decision = request.form.get('final_decision', initial_decision)
        session['responses'].append({
            'ID': task_data['ID'],
            'Condition': condition,
            'Initial_Decision': initial_decision,
            'Final_Decision': final_decision,
            'Predicted': task_data['Predicted'],
            'Confidence_Score': task_data['Confidence_Score'],
            'Level': task_data['Level']
        })
        session['task_index'] += 1
        return redirect(url_for('task'))
    
    customer_info = {
        "RevolvingUtilizationOfUnsecuredLines": task_data["RevolvingUtilizationOfUnsecuredLines"],
        "NumberOfTime30-59DaysPastDueNotWorse": task_data["NumberOfTime30-59DaysPastDueNotWorse"],
        "DebtRatio": task_data["DebtRatio"],
        "MonthlyIncome": task_data["MonthlyIncome"]
    }
    return render_template('stage1.html', customer_info=customer_info, averages=AVERAGES)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
