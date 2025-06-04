from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2 import pool
import uuid
import json
import random
import time

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')

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

# Load updated CSV
CSV_URL = "https://drive.google.com/uc?id=1l87W3PeMpVR1O19BIurIgC9kyqgLvRLr"
df = pd.read_csv(CSV_URL)
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype(int)
    elif df[col].dtype == 'float64':
        df[col] = df[col].astype(float)

# Define average values
AVERAGES = {
    "RevolvingUtilizationOfUnsecuredLines": "46%",
    "NumberOfTime30-59DaysPastDueNotWorse": 0.54,
    "DebtRatio": "43%",
    "MonthlyIncome": 6340
}

# Participant counts for conditions (1, 2, 3)
PARTICIPANT_COUNTS = {1: 0, 2: 0, 3: 0}
MAX_PER_CONDITION = 50

def epistemic_charts(uncertainty_level, total_states=10):
    num_trained_states = 5 + (uncertainty_level - 1)
    random.seed(num_trained_states)
    state_numbers = list(range(1, total_states + 1))
    trained_state_indices = random.sample(state_numbers, num_trained_states)
    
    labels = [f"State {i}" for i in state_numbers]
    heights = [1.0 if i in trained_state_indices else 0.1 for i in state_numbers]
    colors = ['rgba(144, 238, 144, 0.5)' if i in trained_state_indices else 'rgba(169, 169, 169, 0.5)' for i in state_numbers]
    border_colors = ['rgba(0, 128, 0, 1)' if i in trained_state_indices else 'rgba(105, 105, 105, 1)' for i in state_numbers]
    
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Data Volume",
                "data": heights,
                "backgroundColor": colors,
                "borderColor": border_colors,
                "borderWidth": 1.5
            }]
        },
        "options": {
            "scales": {
                "x": {"display": True},
                "y": {"beginAtZero": True, "max": 1.1, "display": False}
            },
            "plugins": {
                "legend": {
                    "display": True,
                    "labels": {
                        "generateLabels": lambda chart: [
                            {"text": "Trained States (High Data Volume)", "fillStyle": "rgba(144, 238, 144, 0.5)", "strokeStyle": "rgba(0, 128, 0, 1)"},
                            {"text": "Untrained States (Low/No Data Volume)", "fillStyle": "rgba(169, 169, 169, 0.5)", "strokeStyle": "rgba(105, 105, 105, 1)"}
                        ]
                    }
                },
                "title": {"display": True, "text": f"Epistemic Uncertainty (Level {uncertainty_level})"}
            }
        }
    }

def aleatoric_charts(level, focal_score=1):
    counts = {i: 0 for i in range(1, 11)}
    max_offset = 3 if focal_score in [1, 2, 3, 8, 9, 10] else 4
    for off in range(-max_offset, max_offset + 1):
        score = focal_score + off
        if score < 1 or score > 10:
            continue
        dist = abs(off)
        if level == 1:
            base = random.randint(3, 4)
        elif level == 2:
            base = random.randint(4, 5) if dist <= 1 else random.randint(3, 4) if dist == 2 else random.randint(0, 2)
        elif level == 3:
            base = random.randint(6, 7) if dist == 0 else random.randint(4, 5) if dist == 1 else random.randint(2, 3) if dist == 2 else random.randint(0, 1)
        elif level == 4:
            base = random.randint(8, 9) if dist == 0 else random.randint(5, 6) if dist == 1 else random.randint(3, 4) if dist == 2 else random.randint(0, 1)
        else:
            base = random.randint(9, 10) if dist == 0 else random.randint(5, 6) if dist == 1 else random.randint(3, 4) if dist == 2 else random.randint(1, 2) if dist == 3 else random.randint(0, 1)
        counts[score] = base
    
    data_points = []
    for score, num in counts.items():
        for i in range(num):
            data_points.append({"x": score, "y": i, "color": 'rgba(0, 128, 0, 0.7)' if score <= 5 else 'rgba(255, 0, 0, 0.7)'})
    
    return {
        "type": "scatter",
        "data": {
            "datasets": [{
                "label": f"Aleatoric Uncertainty (Level {level}, Focal {focal_score})",
                "data": [{"x": point["x"], "y": point["y"]} for point in data_points],
                "backgroundColor": [point["color"] for point in data_points],
                "pointRadius": 10
            }, {
                "type": "line",
                "label": "Threshold",
                "data": [{"x": 5.5, "y": 0}, {"x": 5.5, "y": max(counts.values()) + 1}],
                "borderColor": "rgba(128, 128, 128, 1)",
                "borderDash": [5, 5],
                "pointRadius": 0
            }]
        },
        "options": {
            "scales": {
                "x": {"min": 0.5, "max": 10.5, "ticks": {"stepSize": 1}, "title": {"display": True, "text": "Risk Score (1-10)"}},
                "y": {"min": -0.5, "max": max(counts.values()) + 1, "display": False}
            },
            "plugins": {
                "title": {"display": True, "text": f"Aleatoric Uncertainty (Level {level})"}
            }
        }
    }

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
    participant_name = request.form.get('participant_name', '').strip() or None
    participant_id = str(uuid.uuid4())
    
    PARTICIPANT_COUNTS[condition] += 1
    
    session['condition'] = condition
    session['participant_id'] = participant_id
    session['participant_name'] = participant_name
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
        max_retries = 3
        for attempt in range(max_retries):
            conn = db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    for response in session.get('responses', []):
                        task_data = next(t for t in session['tasks'] if t['ID'] == response['task_id'])
                        revolving_util = float(str(task_data['RevolvingUtilizationOfUnsecuredLines']).replace('%', '')) / 100 if '%' in str(task_data['RevolvingUtilizationOfUnsecuredLines']) else float(str(task_data['RevolvingUtilizationOfUnsecuredLines'])
                        late_payments = int(task_data['NumberOfTime30-59DaysPastDueNotWorse'])
                        debt_ratio = float(str(task_data['DebtRatio']).replace('%', '')) / 100 if '%' in str(task_data['DebtRatio']) else float(str(task_data['DebtRatio'])
                        monthly_income = float(str(task_data['MonthlyIncome']).replace('$', '').replace(',', '')) if any(c in str(task_data['MonthlyIncome']) for c in ['$', ',']) else float(str(task_data['MonthlyIncome'])
                        cur.execute(
                            "INSERT INTO responses (participant_id, task_number, condition, initial_risk_score, final_risk_score, predicted_risk_score, uncertainty_level, uncertainty_type, actual_risk, "
                            "revolving_utilization, late_payments_30_59, debt_ratio, monthly_income) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (session['participant_id'], response['task_number'], response['condition'], response['initial_risk_score'], 
                             response['final_risk_score'], response['predicted_risk_score'], response['uncertainty_level'], 
                             response['uncertainty_type'], response['actual_risk'], revolving_util, late_payments, debt_ratio, monthly_income)
                        )
                    conn.commit()
                break
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                conn.close()
                db_pool.putconn(conn, close=True)
                return "Database connection failed after retries, please try again", 500
            except Exception as e:
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
        initial_risk_score = request.form.get('initial_risk_score')
        try:
            initial_risk_score = int(initial_risk_score)
            if initial_risk_score < 1 or initial_risk_score > 10:
                raise ValueError
        except (ValueError, TypeError):
            return "Invalid risk score. Please enter a number between 1 and 10.", 400
        session['current_initial_risk_score'] = initial_risk_score
        session['current_initial_decision'] = 'accept' if initial_risk_score < 6 else 'reject'
        return redirect(url_for('stage2'))
    
    customer_number = task_index + 1
    customer_info = {
        "RevolvingUtilizationOfUnsecuredLines": task_data["RevolvingUtilizationOfUnsecuredLines"],
        "NumberOfTime30-59DaysPastDueNotWorse": task_data["NumberOfTime30-59DaysPastDueNotWorse"],
        "DebtRatio": task_data["DebtRatio"],
        "MonthlyIncome": task_data["MonthlyIncome"]
    }
    return render_template('stage1.html', customer_info=customer_info, averages=AVERAGES, customer_number=customer_number)

@app.route('/stage2', methods=['GET', 'POST'])
def stage2():
    task_index = session.get('task_index', 0)
    task_data = session['tasks'][task_index]
    condition = session.get('condition')
    initial_risk_score = session.get('current_initial_risk_score')
    initial_decision = session.get('current_initial_decision')
    
    if request.method == 'POST':
        if condition == 1:
            final_risk_score = request.form.get('final_risk_score')
            try:
                final_risk_score = int(final_risk_score)
                if final_risk_score < 1 or final_risk_score > 10:
                    raise ValueError
            except (ValueError, TypeError):
                return "Invalid final risk score. Please enter a number between 1 and 10.", 400
            session['current_final_risk_score'] = final_risk_score
            session['current_final_decision'] = 'accept' if final_risk_score < 6 else 'reject'
            return redirect(url_for('stage3'))
        else:
            action = request.form.get('action')
            if action == 'continue':
                session['show_ai_prediction'] = True
                return render_template(f'stage2_condition{condition}.html', 
                                     predicted_risk_score=task_data['Risk Score'], 
                                     initial_risk_score=initial_risk_score, 
                                     initial_decision=initial_decision, 
                                     show_ai_prediction=True,
                                     graph_data=json.dumps(generate_chart(task_data)))
            elif action == 'submit_final':
                final_risk_score = request.form.get('final_risk_score')
                try:
                    final_risk_score = int(final_risk_score)
                    if final_risk_score < 1 or final_risk_score > 10:
                        raise ValueError
                except (ValueError, TypeError):
                    return "Invalid final risk score. Please enter a number between 1 and 10.", 400
                session['current_final_risk_score'] = final_risk_score
                session['current_final_decision'] = 'accept' if final_risk_score < 6 else 'reject'
                return redirect(url_for('stage3'))
    
    if 'condition' not in session or 'current_initial_risk_score' not in session:
        return redirect(url_for('index'))
    
    def generate_chart(task_data):
        if task_data['Uncertainty Type'].lower() == 'epistemic':
            return epistemic_charts(task_data['Level'])
        else:
            return aleatoric_charts(task_data['Level'], task_data['Risk Score'])
    
    if condition == 1:
        return render_template('stage2_condition1.html', 
                             predicted_risk_score=task_data['Risk Score'], 
                             initial_risk_score=initial_risk_score, 
                             initial_decision=initial_decision)
    else:
        return render_template(f'stage2_condition{condition}.html', 
                             predicted_risk_score=task_data['Risk Score'], 
                             initial_risk_score=initial_risk_score, 
                             initial_decision=initial_decision, 
                             show_ai_prediction=False,
                             graph_data=json.dumps(generate_chart(task_data)))

@app.route('/stage3', methods=['GET', 'POST'])
def stage3():
    task_index = session.get('task_index', 0)
    task_data = session['tasks'][task_index]
    initial_risk_score = session.get('current_initial_risk_score')
    initial_decision = session.get('current_initial_decision')
    final_risk_score = session.get('current_final_risk_score')
    final_decision = session.get('current_final_decision')
    
    if request.method == 'POST':
        session['responses'].append({
            'ID': task_data['ID'],
            'Task_Number': task_index + 1,
            'Condition': session['condition'],
            'Initial_Risk_Score': initial_risk_score,
            'Final_Risk_Score': final_risk_score,
            'Predicted_Risk_Score': task_data['Risk Score'],
            'Uncertainty_Level': task_data['Level'],
            'Uncertainty_Type': task_data['Uncertainty Type'],
            'Actual_Risk': 'high' if task_data['Creditability'] == 1 else 'low'
        })
        session['task_index'] += 1
        return redirect(url_for('task'))
    
    return render_template('stage3.html',
                         customer_number=task_index + 1,
                         initial_decision=initial_decision,
                         final_decision=final_decision,
                         ai_prediction='accept' if task_data['Risk Score'] < 6 else 'reject',
                         actual_risk='high' if task_data['Creditability'] == 1 else 'low')

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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))