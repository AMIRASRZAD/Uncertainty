from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2 import pool

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '1234')

# Neon connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,
    host=os.environ.get('DB_HOST', 'ep-odd-boat-a5tpi1i2-pooler.us-east-2.aws.neon.tech'),
    port=os.environ.get('DB_PORT', 5432),
    database=os.environ.get('DB_NAME', 'neondb'),
    user=os.environ.get('DB_USER', 'neondb_owner'),
    password=os.environ.get('DB_PASSWORD', 'npg_zR21CxagGdVB'),
    sslmode='require'
)

CSV_URL = "https://drive.google.com/uc?id=1l87W3PeMpVR1O19BIurIgC9kyqgLvRLr"
df = pd.read_csv(CSV_URL)
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype(int)

# Define average values
AVERAGES = {
    "RevolvingUtilizationOfUnsecuredLines": "46%" ,
    "NumberOfTime30-59DaysPastDueNotWorse": 0.54,
    "DebtRatio": "43%",
    "MonthlyIncome": 6340
}

# Graph URLs
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

PARTICIPANT_COUNTS = {1: 0, 2: 0, 3: 0, 4: 0}
MAX_PER_CONDITION = 50

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
import uuid

@app.route('/start', methods=['POST'])
def start():
    available_conditions = [c for c, count in PARTICIPANT_COUNTS.items() if count < MAX_PER_CONDITION]
    if not available_conditions:
        return "Experiment is full!", 403
    
    condition = int(np.random.choice(available_conditions))
    participant_name = request.form.get('participant_name', '').strip() or None  # Optional name
    participant_id = str(uuid.uuid4())  # Unique ID for each participant
    
    # Increment participant count for the condition
    PARTICIPANT_COUNTS[condition] += 1
    
    session['condition'] = condition
    session['participant_id'] = participant_id
    session['participant_name'] = participant_name
    session['tasks'] = sample_rows()
    session['task_index'] = 0
    session['responses'] = []
    
    return redirect(url_for('task'))
import time

@app.route('/task', methods=['GET', 'POST'])
def task():
    if 'condition' not in session:
        print("No condition in session, redirecting to index")
        return redirect(url_for('index'))
    
    task_index = session['task_index']
    if task_index >= len(session['tasks']):
        max_retries = 3
        for attempt in range(max_retries):
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
                            "INSERT INTO responses (participant_id, task_number, condition, initial_decision, final_decision, predicted, confidence_score, level, "
                            "revolving_utilization, late_payments_30_59, debt_ratio, monthly_income) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (session['participant_id'], response['Task_Number'], response['Condition'], response['Initial_Decision'], 
                             response['Final_Decision'], response['Predicted'], response['Confidence_Score'], 
                             response['Level'], revolving_util, late_payments, debt_ratio, monthly_income)
                        )
                    conn.commit()
                    print("Data committed to Neon")
                break  # Exit retry loop on success
            except psycopg2.OperationalError as e:
                print(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                conn.close()
                db_pool.putconn(conn, close=True)
                return "Database connection failed after retries, please try again", 500
            except Exception as e:
                print(f"Database error: {e}")
                conn.rollback()
                db_pool.putconn(conn)
                return "Error saving data", 500
            finally:
                if conn and not conn.closed:
                    db_pool.putconn(conn)
        return render_template('end.html')
    
    task_data = session['tasks'][task_index]
    condition = session['condition']
    
    if request.method == 'POST':
        initial_decision = request.form.get('initial_decision')
        print(f"Task POST - Initial Decision: {initial_decision}")
        if initial_decision not in ['high risk', 'low risk']:
            print("Invalid initial_decision, aborting")
            return "Invalid decision", 400
        session['current_initial_decision'] = initial_decision
        print(f"Session set: current_initial_decision={session['current_initial_decision']}")
        return redirect(url_for('stage2'))
    
    customer_number = task_index + 1
    customer_info = {
        "RevolvingUtilizationOfUnsecuredLines": task_data["RevolvingUtilizationOfUnsecuredLines"],
        "NumberOfTime30-59DaysPastDueNotWorse": task_data["NumberOfTime30-59DaysPastDueNotWorse"],
        "DebtRatio": task_data["DebtRatio"],
        "MonthlyIncome": task_data["MonthlyIncome"]
    }
    print(f"Rendering Stage 1 - Customer {customer_number}")
    return render_template('stage1.html', customer_info=customer_info, averages=AVERAGES, customer_number=customer_number)



@app.route('/stage2', methods=['GET', 'POST'])
def stage2():
    task_index = session.get('task_index', 0)
    task_data = session['tasks'][task_index]
    condition = session.get('condition')
    initial_decision = session.get('current_initial_decision')
    
    print(f"Stage 2 - Method: {request.method}, Condition: {condition}, Initial Decision: {initial_decision}")
    
    if request.method == 'POST':
        final_decision = request.form.get('final_decision')
        print(f"POST received - Final Decision: {final_decision}")
        if final_decision not in ['high risk', 'low risk']:
            print(f"Invalid final_decision, defaulting to {initial_decision}")
            final_decision = initial_decision or 'Not Set'
        session['current_final_decision'] = final_decision
        print("Redirecting to stage3")
        return redirect(url_for('stage3'))
    
    if 'condition' not in session or 'current_initial_decision' not in session:
        print("Session missing condition or initial_decision on GET, redirecting to task")
        return redirect(url_for('task'))
    
    print("Rendering Stage 2 template")
    if condition == 1:
        return render_template('stage2_condition1.html', predicted=task_data['Predicted'], initial_decision=initial_decision)
    elif condition == 2:
        return render_template('stage2_condition2.html', predicted=task_data['Predicted'], confidence_score=task_data['Confidence_Score'], initial_decision=initial_decision)
    elif condition == 3:
        level_map = {1: "0.5-0.6", 2: "0.6-0.7", 3: "0.7-0.8", 4: "0.8-0.9", 5: "0.9-1.0"}
        graph_url = GRAPH_URLS["condition3"][level_map[task_data['Level']]]
        return render_template('stage2_condition3.html', predicted=task_data['Predicted'], confidence_score=task_data['Confidence_Score'], graph_url=graph_url, initial_decision=initial_decision)
    elif condition == 4:
        level_map = {1: "0.5-0.6", 2: "0.6-0.7", 3: "0.7-0.8", 4: "0.8-0.9", 5: "0.9-1.0"}
        predicted_lower = task_data['Predicted'].lower()
        outcome = "default" if "high" in predicted_lower else "non-default"
        key = f"{level_map[task_data['Level']]}-{outcome}"
        graph_key = level_map[task_data['Level']] if task_data['Level'] == 1 else key
        graph_url = GRAPH_URLS["condition4"][graph_key]
        return render_template('stage2_condition4.html', predicted=task_data['Predicted'], confidence_score=task_data['Confidence_Score'], graph_url=graph_url, initial_decision=initial_decision)
       

@app.route('/stage3', methods=['GET', 'POST'])
def stage3():
    task_index = session['task_index']
    task_data = session['tasks'][task_index]
    initial_decision = session.get('current_initial_decision', 'Not Set')
    final_decision = session.get('current_final_decision', 'Not Set')
    
    # Get actual risk and AI prediction
    actual_risk = "High Risk" if task_data['Creditability'] == 1 else "Low Risk"
    ai_prediction = task_data['Predicted']
    
    if request.method == 'POST':
        session['responses'].append({
            'ID': task_data['ID'],
            'Task_Number': task_index + 1,
            'Condition': session['condition'],
            'Initial_Decision': initial_decision,
            'Final_Decision': final_decision,
            'Predicted': task_data['Predicted'],
            'Confidence_Score': task_data['Confidence_Score'],
            'Level': task_data['Level']
        })
        session['task_index'] += 1
        return redirect(url_for('task'))
    
    customer_number = task_index + 1
    return render_template('stage3.html', 
                          customer_number=customer_number,
                          initial_decision=initial_decision,
                          final_decision=final_decision,
                          ai_prediction=ai_prediction,
                          actual_risk=actual_risk)

@app.route('/test-db')
def test_db():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT NOW();")
            result = cur.fetchone()
            return f"Database time: {result[0]}"
    except Exception as e:
        return f"Connection failed: {e}"
    finally:
        db_pool.putconn(conn)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=True)
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
