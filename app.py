# app.py
# Personal Finance Manager - Single-file Flask app
# Requirements: Flask, pandas, sklearn, matplotlib (for server-side image generation optional)
# We'll use SQLite (builtin) and Chart.js on the frontend.

from flask import Flask, g, render_template_string, request, redirect, url_for, jsonify
import sqlite3
import os
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import json

DB = 'finance.db'
app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-your-own-secret'

# ---------- DB helpers ----------
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB, detect_types=sqlite3.PARSE_DECLTYPES)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL NOT NULL,
        category TEXT NOT NULL,
        note TEXT,
        date DATE NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        k TEXT PRIMARY KEY,
        v TEXT
    );
    """)
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ---------- Basic routes ----------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Personal Finance Manager</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{font-family:Inter,Segoe UI,Arial;background:#0f1724;color:#e6eef8;margin:0;padding:0}
    .container{max-width:1100px;margin:30px auto;padding:20px}
    .card{background:linear-gradient(145deg,#0b1220,#0f1b2a);padding:18px;border-radius:12px;box-shadow:0 6px 22px rgba(2,6,23,0.6);margin-bottom:18px}
    h1{margin:0 0 8px}
    form input, form select{padding:8px;margin:6px 6px 6px 0;border-radius:6px;border:1px solid #233047;background:#08121b;color:#e6eef8}
    .row{display:flex;flex-wrap:wrap;gap:8px}
    button{background:#06b6d4;color:#002; padding:10px 14px;border-radius:8px;border:none;cursor:pointer}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;text-align:left;border-bottom:1px solid #233047}
    .small{font-size:0.9rem;color:#9fb0c9}
    canvas{max-width:100%}
    .muted{color:#84a0bc;font-size:0.9rem}
    .topline{display:flex;justify-content:space-between;align-items:center}
    .actions{display:flex;gap:8px}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <div class="container">
    <div class="topline">
      <div>
        <h1>Personal Finance Manager</h1>
        <div class="small">Track expenses • Analyze spending • Get ML suggestions</div>
      </div>
      <div class="actions">
        <form method="post" action="/set-income" style="display:flex;gap:8px;align-items:center">
          <input name="monthly_income" placeholder="Monthly income (₹)" type="number" step="1" required value="{{ income or '' }}">
          <button type="submit">Set Income</button>
        </form>
      </div>
    </div>

    <div class="card">
      <h3>Add expense</h3>
      <form method="post" action="/add" class="row">
        <input name="amount" placeholder="Amount (₹)" required type="number" step="0.01">
        <select name="category" required>
          <option value="Food">Food</option>
          <option value="Transport">Transport</option>
          <option value="Bills">Bills</option>
          <option value="Shopping">Shopping</option>
          <option value="Entertainment">Entertainment</option>
          <option value="Other">Other</option>
        </select>
        <input name="note" placeholder="Note (optional)">
        <input name="date" type="date" value="{{ today }}" required>
        <button type="submit">Add</button>
      </form>
    </div>

    <div class="card">
      <h3>Recent expenses</h3>
      <table>
        <thead><tr><th>When</th><th>Amount</th><th>Category</th><th>Note</th><th></th></tr></thead>
        <tbody>
          {% for e in expenses %}
          <tr>
            <td>{{ e['date'] }}</td>
            <td>₹{{ '{:.2f}'.format(e['amount']) }}</td>
            <td>{{ e['category'] }}</td>
            <td class="muted">{{ e['note'] or '' }}</td>
            <td><a href="/delete/{{ e['id'] }}" style="color:#ff7b7b">Delete</a></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h3>Spending analysis</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px">
        <div>
          <canvas id="byCategory"></canvas>
        </div>
        <div>
          <canvas id="timeseries"></canvas>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>ML Recommendations</h3>
      <div id="recommendation" class="small">Loading...</div>
      <button onclick="fetchRec()">Refresh Recommendation</button>
    </div>

    <div class="card small">
      <div>Export/backup: download JSON of all expenses</div>
      <a href="/export" style="color:#bfe9ff">Download data (JSON)</a>
    </div>

  </div>

<script>
async function fetchData(){
  const r = await fetch('/data');
  const j = await r.json();
  drawCharts(j);
  showRec(j.recommendation_text || 'No recommendation yet. Click refresh.');
}
function showRec(txt){
  document.getElementById('recommendation').innerText = txt;
}

function drawCharts(payload){
  const byCat = payload.by_category || {};
  const labels = Object.keys(byCat);
  const values = Object.values(byCat).map(v=>+v.toFixed(2));

  const ts = payload.timeseries || {};
  const ts_labels = Object.keys(ts);
  const ts_values = Object.values(ts).map(v=>+v.toFixed(2));

  // byCategory
  const ctx = document.getElementById('byCategory').getContext('2d');
  if(window._catChart) window._catChart.destroy();
  window._catChart = new Chart(ctx, {
    type:'doughnut',
    data:{labels, datasets:[{data:values, label: 'By category'}]},
    options:{plugins:{legend:{position:'bottom'}}}
  });

  const ctx2 = document.getElementById('timeseries').getContext('2d');
  if(window._tsChart) window._tsChart.destroy();
  window._tsChart = new Chart(ctx2, {
    type:'line',
    data:{labels:ts_labels, datasets:[{data:ts_values, label:'Monthly total', fill:false, tension:0.3}]},
    options:{scales:{y:{beginAtZero:true}}}
  });
}

async function fetchRec(){
  const r = await fetch('/recommend');
  const j = await r.json();
  showRec(j.text);
}

fetchData();
</script>

</body>
</html>
"""

@app.route('/')
def index():
    init_db()
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM expenses ORDER BY date DESC LIMIT 50")
    expenses = cur.fetchall()
    # read income meta
    cur.execute("SELECT v FROM meta WHERE k='monthly_income'")
    row=cur.fetchone()
    income = float(row['v']) if row else None
    return render_template_string(INDEX_HTML, expenses=expenses, today=datetime.today().strftime('%Y-%m-%d'), income=income)

@app.route('/add', methods=['POST'])
def add():
    amount = float(request.form['amount'])
    category = request.form['category']
    note = request.form.get('note','')
    date = request.form['date']
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO expenses (amount, category, note, date) VALUES (?,?,?,?)",
               (amount, category, note, date))
    db.commit()
    return redirect(url_for('index'))

@app.route('/delete/<int:_id>')
def delete(_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM expenses WHERE id=?", (_id,))
    db.commit()
    return redirect(url_for('index'))

@app.route('/export')
def export():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM expenses ORDER BY date")
    rows = cur.fetchall()
    arr = [dict(row) for row in rows]
    return jsonify(arr)

@app.route('/set-income', methods=['POST'])
def set_income():
    income = request.form.get('monthly_income')
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT OR REPLACE INTO meta (k,v) VALUES ('monthly_income',?)", (str(income),))
    db.commit()
    return redirect(url_for('index'))

# ---------- Data API used by charts and ML ----------
def df_from_db():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT date, amount, category FROM expenses ORDER BY date")
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=['date','amount','category'])
    df = pd.DataFrame(rows, columns=['date','amount','category'])
    df['date'] = pd.to_datetime(df['date'])
    return df

@app.route('/data')
def data_api():
    df = df_from_db()
    # by category totals
    by_cat = df.groupby('category')['amount'].sum().to_dict()
    # timeseries: month totals YYYY-MM
    if not df.empty:
        df['ym'] = df['date'].dt.to_period('M').astype(str)
        ts = df.groupby('ym')['amount'].sum().to_dict()
    else:
        ts = {}
    # basic recommendation text stored in meta (fast)
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT v FROM meta WHERE k='last_recommendation'")
    row = cur.fetchone()
    rec_text = row['v'] if row else ''
    return jsonify({'by_category': by_cat, 'timeseries': ts, 'recommendation_text': rec_text})

@app.route('/recommend')
def recommend():
    """
    ML-powered recommendation endpoint.
    1) Predict next month's total expense using linear regression on monthly totals (if >=3 months)
    2) Compute suggested savings target:
         - If predicted expense < income: suggest saving at least 20% of income or (income - predicted expense) whichever is smaller/larger
         - If predicted expense > income: suggest cutting discretionary categories by X% and prioritize emergency fund
    3) Suggest a simple investment split based on user's spending mix.
    """
    df = df_from_db()
    db = get_db(); cur = db.cursor()
    cur.execute("SELECT v FROM meta WHERE k='monthly_income'")
    row = cur.fetchone()
    income = float(row['v']) if row else None

    if df.empty:
        text = "No expenses recorded yet — add some expenses and then click Recommendations."
        cur.execute("INSERT OR REPLACE INTO meta (k,v) VALUES ('last_recommendation',?)", (text,))
        db.commit()
        return jsonify({'text': text})

    df['ym'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('ym')['amount'].sum().reset_index()
    monthly['index'] = np.arange(len(monthly))  # 0,1,2...
    text_lines = []

    # prediction
    if len(monthly) >= 3:
        X = monthly[['index']].values
        y = monthly['amount'].values
        model = LinearRegression()
        model.fit(X, y)
        next_idx = np.array([[monthly['index'].max() + 1]])
        pred = float(model.predict(next_idx)[0])
        pred = max(0.0, pred)
        text_lines.append(f"Predicted next-month spending: ₹{pred:,.2f} (based on {len(monthly)} months).")
    else:
        pred = float(monthly['amount'].mean())
        text_lines.append(f"Estimated next-month spending (insufficient history; using average): ₹{pred:,.2f}.")

    # savings suggestion
    if income is None:
        text_lines.append("Set your monthly income at the top to get tailored savings suggestions.")
    else:
        # simple rule-based suggestion blending with prediction
        spare = income - pred
        if spare > 0:
            recommended_saving = max(0.1*income, min(0.4*income, spare*0.7))  # keep some buffer
            text_lines.append(f"Recommended monthly savings: ₹{recommended_saving:,.2f} (~{recommended_saving/income*100:.0f}% of income).")
        else:
            # over-spending risk
            text_lines.append("Warning: predicted spending >= income. Reduce discretionary categories by 10-30% and prioritize urgent bills/emergency fund.")

    # category-based suggestions
    by_cat = df.groupby('category')['amount'].sum()
    total = by_cat.sum()
    if total > 0:
        top = by_cat.sort_values(ascending=False).head(3)
        text_lines.append("Top spending categories: " + ", ".join([f"{c} (₹{v:,.0f})" for c,v in top.items()]) + ".")
        # simple investment allocation heuristic:
        # if Food/Transport high -> recommend 60% emergency+debt paydown, else if Bills high -> 50% emergency 30% SIP etc.
        pct_food = by_cat.get('Food', 0) / total
        pct_bills = by_cat.get('Bills', 0) / total
        if pct_food > 0.35:
            alloc = "Emergency fund 60%, Debt/High-interest paydown 20%, SIP/Invest 20%."
        elif pct_bills > 0.3:
            alloc = "Emergency fund 50%, SIP (equity) 30%, Debt/Fixed income 20%."
        else:
            alloc = "Emergency fund 30%, SIP (equity) 40%, Fixed income 30%."
        text_lines.append("Suggested allocation: " + alloc)
    else:
        text_lines.append("No category data to compute allocations.")

    text = " ".join(text_lines)
    # save brief text to meta for dashboard use
    cur.execute("INSERT OR REPLACE INTO meta (k,v) VALUES ('last_recommendation',?)", (text,))
    db.commit()
    return jsonify({'text': text})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


