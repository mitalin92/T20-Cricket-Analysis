

## 🏏 T20 IPL Cricket Analysis (2018 to 2024)

An interactive Streamlit-based data science project that analyzes batter performance and dismissal patterns in the Indian Premier League (IPL) from 2018 to 2024.

Built by Sai Arun, Mustafa, and Mitali, this tool uses ball-by-ball data to generate tactical insights for cricket fans, analysts, and teams. The dataset includes detailed information such as shot direction, line and length, dismissal types, control, bowling style, and more.

## 📌 Features

Boundary Scoring Visualizations

    General Wagon Wheel: Fixed-length shot visuals for all 4s and 6s

    Intelligent Wagon Wheel: Adjusts shot length dynamically based on shot difficulty

    Wagon Zone Wheel: Maps scoring zones (1 to 8) with strike rate, run contributions, and dismissal risk

Shot and Dismissal Analysis

    Line and Length Heatmaps: Shows dismissal likelihood based on delivery trajectory

    Shot Difficulty Score: Visualizes batter performance across high and low difficulty shots

    Control vs Non-Control: Compares boundaries and dismissals for controlled vs uncontrolled shots

Match-Up and Tactical Insights

    Bowling Style Breakdown: Compare performance against specific bowling styles (e.g., RFM, OB, SLA) using strike rate, boundary percentage, and impact per 100 balls

    Spin vs Pace Analysis: Zone-wise comparison of batting performance

    Pressure Handling: How batters perform after consecutive dot balls

    Strategic Takeaways: Highlights weaknesses and strengths for both batters and bowlers

## 📊 Dataset

Source

    IPL_2018_2024.xlsx (ball-by-ball data)

Key Variables

    bat, batruns, dismissal, out

    line, length, wagonX, wagonY, wagonZone

    control, bowl_style, bat_hand, year

  ##  💻 Project Structure

Introduction

    Overview of T20 cricket and the IPL format

Data Analysis Tabs

    Interactive visualizations and metrics across 7 sub-tabs

Help Tab

    Explanation of each variable used

Sidebar Filters

    Select batter

    Filter by year range

    Choose bowler type: All, Spin, or Pace

    Toggle control shots on or off

  ##  👥 Contributors

Mitali, Mustafa, Arun 

Part of the Data Science Lab Capstone Project
