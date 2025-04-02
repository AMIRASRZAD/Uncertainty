from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2 import pool

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '1234')

# Supabase DB connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,
    host=os.environ.get('DB_HOST', 'db.lyqtayprcpyfjhdmlluk.supabase.co'),
    port=os.environ.get('DB_PORT', 5432),
    database=os.environ.get('DB_NAME', 'postgres'),
    user=os.environ.get('DB_USER', 'postgres'),
    password=os.environ.get('DB_PASSWORD', 'DhAaQV4tM$K!!gr')
)

CSV_URL = "https://drive.google.com/uc?id=1EV4AoEymcBA3FEFgce2-Dq9cBSXu4rIu"
df = pd.read_csv(CSV_URL)
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype(int)
        

# Define average values
AVERAGES = {
    "RevolvingUtilizationOfUnsecuredLines": .25,
    "NumberOfTime30-59DaysPastDueNotWorse": 0.1,
    "DebtRatio": .20,
    "MonthlyIncome": 3000
}

# Graph URLs (updated with &export=download)
GRAPH_URLS = {
    "condition3": {
        "0.5-0.6": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition3-level%200.5-0.6.png",
        "0.6-0.7": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition3-level%200.6-0.7.png",
        "0.7-0.8": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition3-level%200.7-0.8.png",
        "0.8-0.9": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition3-level%200.8-0.9.png",
        "0.9-1.0": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition3-level%200.9-1.0.png"
    },
    "condition4": {
        "0.5-0.6": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.5-0.6.png",
        "0.6-0.7-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.6-0.7-default.png",
        "0.6-0.7-non-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.6-0.7-non-default.png",
        "0.7-0.8-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.7-0.8-default.png",
        "0.7-0.8-non-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.7-0.8-non-default.png",
        "0.8-0.9-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.8-0.9-default.png",
        "0.8-0.9-non-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.8-0.9-non-default.png",
        "0.9-1.0-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.9-1.0-default.png",
        "0.9-1.0-non-default": "https://raw.githubusercontent.com/AMIRASRZAD/Uncertainty/main/condition4-level%200.9-1.0-non-default.png"
    }
}

# Condition assignment tracking
PARTICIPANT_COUNTS = {1: 0, 2: 0, 3: 0, 4: 0}
MAX_PER_CONDITION = 50

# Sample 20 rows per participant with explicit type conversion
def sample_rows():
    ranges = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    sampled_rows = []
    for r in ranges:
        subset = df[df["Level"] == ranges.index(r) + 1].sample(4, random_state=np.random.randint(1000))
        for row in subset.to_dict("records"):
            converted_row = {}
            for key, value in row.items():
                if isinstance(value, np.int64):
                    converted_row[key] = int(value)
                elif isinstance(value, np.float64):
                    converted_row[key] = float(value)
                else:
                    converted_row[key] = value
            sampled_rows.append(converted_row)
    np.random.shuffle(sampled_rows)
    return sampled_rows[:20]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    available_conditions = [c for c, count in PARTICIPANT_COUNTS.items() if count < MAX_PER_CONDITION]
    if not available_conditions:
        return "Experiment is full!", 403
    condition = int(np.random.choice(available_conditions))
    
    session['condition'] = condition
    session['tasks'] = sample_rows()
    session['task_index'] = 0
    session['responses'] = []
    
    return redirect(url_for('task'))

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
                    # Convert percentage strings to floats
                    revolving_util = float(task_data['RevolvingUtilizationOfUnsecuredLines'].replace('%', '')) / 100 if isinstance(task_data['RevolvingUtilizationOfUnsecuredLines'], str) and '%' in task_data['RevolvingUtilizationOfUnsecuredLines'] else float(task_data['RevolvingUtilizationOfUnsecuredLines'])
                    debt_ratio = float(task_data['DebtRatio'].replace('%', '')) / 100 if isinstance(task_data['DebtRatio'], str) and '%' in task_data['DebtRatio'] else float(task_data['DebtRatio'])
                    monthly_income = float(task_data['MonthlyIncome'].replace('$', '').replace(',', '')) if isinstance(task_data['MonthlyIncome'], str) else float(task_data['MonthlyIncome'])
                    cur.execute(
                        "INSERT INTO responses (participant_id, condition, initial_decision, final_decision, predicted, confidence_score, level, "
                        "revolving_utilization, late_payments_30_59, debt_ratio, monthly_income) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (response['ID'], response['Condition'], response['Initial_Decision'], 
                         response['Final_Decision'], response['Predicted'], response['Confidence_Score'], 
                         response['Level'], revolving_util, 
                         task_data['NumberOfTime30-59DaysPastDueNotWorse'], debt_ratio, 
                         monthly_income)
                    )
                conn.commit()
                print("Data committed to Supabase")
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


@app.route('/stage2', methods=['POST'])
def stage2():
    initial_decision = request.form['initial_decision']
    task_data = session['tasks'][session['task_index']]
    condition = session['condition']
    
    if condition == 1:
        return render_template('stage2_condition1.html', 
                              predicted=task_data['Predicted'], 
                              initial_decision=initial_decision)
    elif condition == 2:
        return render_template('stage2_condition2.html', 
                              predicted=task_data['Predicted'], 
                              confidence_score=task_data['Confidence_Score'], 
                              initial_decision=initial_decision)
    elif condition == 3:
        level_map = {1: "0.5-0.6", 2: "0.6-0.7", 3: "0.7-0.8", 4: "0.8-0.9", 5: "0.9-1.0"}
        graph_url = GRAPH_URLS["condition3"][level_map[task_data['Level']]]
        return render_template('stage2_condition3.html', 
                              predicted=task_data['Predicted'], 
                              confidence_score=task_data['Confidence_Score'], 
                              graph_url=graph_url, 
                              initial_decision=initial_decision)
    elif condition == 4:
        level_map = {1: "0.5-0.6", 2: "0.6-0.7", 3: "0.7-0.8", 4: "0.8-0.9", 5: "0.9-1.0"}
        key = f"{level_map[task_data['Level']]}-{task_data['Predicted'].lower()}"
        graph_key = level_map[task_data['Level']] if task_data['Level'] == 1 else key
        graph_url = GRAPH_URLS["condition4"][graph_key]
        return render_template('stage2_condition4.html', 
                              predicted=task_data['Predicted'], 
                              confidence_score=task_data['Confidence_Score'], 
                              graph_url=graph_url, 
                              initial_decision=initial_decision)

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
