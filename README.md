# T20-Cricket IPL Analysis (2018–2024)

An interactive Streamlit-based data science project analyzing batter performance and dismissal patterns in the Indian Premier League (IPL) T20 competition from 2018 to 2024.

The tool leverages detailed ball-by-ball data—including line, length, control, wagon wheel coordinates, and dismissal types—to provide actionable insights for cricket enthusiasts, analysts, and teams.

---

## 🌟 Key Features

### 🔹 Boundary Scoring Visualizations
- **General Wagon Wheel:** Fixed-length shot visuals for all 4s and 6s
- **Intelligent Wagon Wheel:** Adjusts shot length dynamically based on shot difficulty
- **Wagon Zone Wheel:** Maps scoring zones (1 to 8) with strike rate, run contributions, and dismissal risk

### 🔹 Shot and Dismissal Analysis
- **Line and Length Heatmaps:** Shows dismissal likelihood based on delivery trajectory
- **Shot Difficulty Score:** Visualizes batter performance across high and low difficulty shots
- **Control vs. Non-Control:** Compares boundaries and dismissals for controlled vs. uncontrolled shots

### 🔹 Match-Up and Tactical Insights
- **Bowling Style Matchups:** Compares batter performance against specific bowling styles (e.g., RFM, OB)
- **Batting Hand vs Bowling Type:** LHB vs RHB comparisons across bowling styles
- **Phase-Based Dismissals:** Powerplay, middle overs, and death overs impact analysis

---

## 📚 Dataset

**Source:** IPL_2018_2024.xlsx (ball-by-ball data)

**Key Variables:**
- `bat`: Batter name
- `batruns`: Runs per delivery
- `out`, `dismissal`: Wicket status and dismissal type
- `wagonX`, `wagonY`, `wagonZone`: Shot coordinates and zones
- `line`, `length`: Delivery trajectory
- `control`: Controlled vs. uncontrolled shot
- `bowl_style`, `bowl_kind`: Bowling category
- `bat_hand`: Left or right-handed
- `year`, `phase`, `team_bowl`, `team_bat`

---

## 🌐 Project Structure

### Tabs:
- **Introduction:** Overview of cricket, T20, and IPL
- **Data Analysis:** 7 sub-tabs with visualizations and tactical suggestions
- **Help:** Explanation of all variables and filters

### Sidebar Filters:
- Batter selector
- Year range slider (2018–2024)
- Bowler type toggle (All / Spin / Pace)
- Shot control filter

---

## 📊 Requirements
- Python 3.8+
- Libraries: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---

## 👨‍💼 Contributors

**Mitali**, **Mustafa**, and **Arun**  
_Data Science Lab Project_

---

## ✨ Try it out
Use the sidebar to:
- Select a player (e.g. Virat Kohli)
- Filter years (e.g. 2020–2024)
- Compare across phases (PP, Middle, Death)
- View heatmaps, tactical tables, and personalized matchups!

---

