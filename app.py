import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns

@st.cache_data
def ld(path="IPL_2018_2024.xlsx"):
    df = pd.read_excel(path)
    cols = [
        "bat", "batruns", "out", "dismissal",
        "wagonX", "wagonY", "wagonZone", "line", "length",
        "bowl_style", "year", "bat_hand"
    ]
    df = df[[c for c in cols if c in df.columns]].drop_duplicates()
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    spin = {"OB", "LB", "LBG", "SLA", "LWS", "RWS"}
    pace = {"RF", "RFM", "RMF", "RM", "LF", "LFM", "LMF", "LM"}
    def classify_bowler(s):
        if s in spin:
            return "Spin"
        elif s in pace:
            return "Pace"
        else:
            return "Unknown"
    if "bowl_style" in df.columns:
        df["bowler_type"] = df["bowl_style"].apply(classify_bowler)
    else:
        df["bowler_type"] = "Unknown"
    return df

def shift_coords(df):
    centerX = df["wagonX"].median()
    centerY = df["wagonY"].median()
    df["plotX"] = df["wagonX"] - centerX
    df["plotY"] = df["wagonY"] - centerY
    return df

def sd(df):
    zone_ct = df.groupby(["line", "length", "wagonZone"]).size().reset_index(name="ShotsInZone")
    total_ct = df.groupby(["line", "length"]).size().reset_index(name="AllShots")
    merged = pd.merge(zone_ct, total_ct, on=["line", "length"], how="left")
    def calc_sd(row):
        if row["ShotsInZone"] == 0:
            return 1.0
        return row["AllShots"] / row["ShotsInZone"]
    merged["shot_difficulty"] = merged.apply(calc_sd, axis=1)
    df = pd.merge(
        df,
        merged[["line", "length", "wagonZone", "shot_difficulty"]],
        on=["line", "length", "wagonZone"],
        how="left"
    )
    df["shot_difficulty"] = df["shot_difficulty"].fillna(1.0)
    return df

def wagonGen(df, ax):
    bdf = df[df["batruns"].isin([4,6])].copy()
    if bdf.empty:
        ax.set_title("Normal Wagon Wheel - No Boundaries")
        ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
        ax.set_xlim(-220,220)
        ax.set_ylim(-220,220)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        return bdf
    ax.set_title("General Boundary Wagon Wheel")
    ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
    bdf["angle_rad"] = np.arctan2(bdf["plotY"], bdf["plotX"])
    bdf["fixedX"] = 200 * np.cos(bdf["angle_rad"])
    bdf["fixedY"] = 200 * np.sin(bdf["angle_rad"])
    for _, row in bdf.iterrows():
        color = "green" if row["batruns"] == 4 else "purple"
        lw = 1.0 if row["batruns"] == 4 else 1.5
        ax.plot([0, row["fixedX"]], [0, row["fixedY"]], color=color, linewidth=lw, alpha=0.8)
    ax.set_xlim(-220,220)
    ax.set_ylim(-220,220)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="4 runs"),
        Line2D([0], [0], color="purple", lw=2, label="6 runs")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    return bdf

def wagonSD(df, ax):
    bdf = df[df["batruns"].isin([4,6])].copy()
    if bdf.empty:
        ax.set_title("Intelligent Wagon Wheel - No Boundaries")
        ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
        ax.set_xlim(-220,220)
        ax.set_ylim(-220,220)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        return bdf
    ax.set_title("Intelligent Wagon Wheel")
    ax.add_patch(Circle((0,0),200,color="green",alpha=0.2))
    bdf["runs_pos"] = bdf["batruns"].clip(lower=0)
    bdf["magnitude_raw"] = bdf["runs_pos"] * bdf["shot_difficulty"]
    bdf["angle_rad"] = np.arctan2(bdf["plotY"], bdf["plotX"])
    max_mag = bdf["magnitude_raw"].max()
    scale_factor = 200 / max_mag if max_mag > 0 else 1
    bdf["scaled_mag"] = bdf["magnitude_raw"] * scale_factor
    bdf["intX"] = bdf["scaled_mag"] * np.cos(bdf["angle_rad"])
    bdf["intY"] = bdf["scaled_mag"] * np.sin(bdf["angle_rad"])
    for _, row in bdf.iterrows():
        color = "green" if row["batruns"] == 4 else "purple"
        lw = 1.0 if row["batruns"] == 4 else 1.5
        ax.plot([0, row["intX"]], [0, row["intY"]], color=color, linewidth=lw, alpha=0.8)
    ax.set_xlim(-220,220)
    ax.set_ylim(-220,220)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="4 runs"),
        Line2D([0], [0], color="purple", lw=2, label="6 runs")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    return bdf

def wagonZoneWheel(df, ax):
    df = df[df["wagonZone"] != 0].copy()
    if df.empty:
        ax.set_title("Wagon Zone Wheel - No Valid Zones")
        return pd.DataFrame()
    def zone_stats(sub):
        balls = len(sub)
        runs = sub["batruns"].sum()
        outs = sub["dismissal"].notna().sum()
        sr = 100 * runs / balls if balls > 0 else 0
        ave = runs / outs if outs > 0 else float("inf")
        return pd.Series({"Balls": balls, "Runs": runs, "SR": sr, "Ave": ave})
    
    zone_df = df.groupby("wagonZone").apply(zone_stats).reset_index()
    zone_df["SR"] = zone_df["SR"].round(1)
    zone_df["Ave"] = zone_df["Ave"].apply(lambda x: round(x,1) if math.isfinite(x) else float("inf"))
    total_runs = zone_df["Runs"].sum()
    zone_df["pct_runs"] = 100 * zone_df["Runs"] / total_runs if total_runs > 0 else 0
    max_zone_runs = zone_df["Runs"].max() if not zone_df.empty else 1
    scf = 0.6
    outer_r = 0.8

    for i in range(1, 9):
        row = zone_df[zone_df["wagonZone"] == i]
        if row.empty:
            sr, ave, balls, runs, pctruns = 0, 0, 0, 0, 0
        else:
            sr = row["SR"].values[0]
            ave = row["Ave"].values[0]
            balls = row["Balls"].values[0]
            runs = row["Runs"].values[0]
            pctruns = row["pct_runs"].values[0]

        ang = {
            1: (45, 90),
            2: (0, 45),
            3: (315, 360),
            4: (270, 315),
            5: (225, 270),
            6: (180, 225),
            7: (135, 180),
            8: (90, 135)
        }
        start_angle, end_angle = ang.get(i, (0, 0))
        wedge_green = Wedge((0, 0), outer_r, start_angle, end_angle,
                            facecolor="limegreen", alpha=0.5, edgecolor="gray")
        ax.add_patch(wedge_green)
        frac = (runs / max_zone_runs) * scf if max_zone_runs > 0 else 0
        wedge_blue = Wedge((0, 0), frac, start_angle, end_angle,
                           facecolor="blue", alpha=0.5)
        ax.add_patch(wedge_blue)
        
        mid_angle = (start_angle + end_angle) / 2
        mid_rad = math.radians(mid_angle)
        text_r = 0.55
        text_x = text_r * math.cos(mid_rad)
        text_y = text_r * math.sin(mid_rad)
        ave_str = "inf" if ave == float("inf") else f"{ave}"
        zone_text = (f"Zone {i}\n{sr} SR\n{balls}b\n{pctruns:.1f}% runs")
        ax.text(text_x, text_y, zone_text, ha="center", va="center", fontsize=8, color="black")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("Wagon Zone Wheel")
    return zone_df

def tab(df, df_global):
    if "bowl_style" not in df.columns:
        return pd.DataFrame()
    def style_metrics(g):
        b = len(g)
        r = g["batruns"].sum()
        d = g["dismissal"].notna().sum()
        boundary_ct = g[g["batruns"].isin([4,6])].shape[0]
        dot_ct = g[g["batruns"]==0].shape[0]
        sr_val = 100 * r / b if b > 0 else 0
        dismiss_pct = d
        boundary_pct = 100 * boundary_ct / b if b > 0 else 0
        dot_pct = 100 * dot_ct / b if b > 0 else 0
        ave_val = r / d if d > 0 else float("inf")
        return pd.Series({
            "Balls": b,
            "Runs": r,
            "SR": sr_val,
            "Dismissals": dismiss_pct,
            "Boundary%": boundary_pct,
            "Dot%": dot_pct,
            "Ave": ave_val,
            "Dismissals_count": d
        })
    local_df = df.groupby("bowl_style").apply(style_metrics).reset_index()
    gf = df_global.groupby("bowl_style").agg(
        total_runs=("batruns","sum"),
        total_balls=("batruns","count")
    ).reset_index()
    gf["global_SR"] = 100 * gf["total_runs"] / gf["total_balls"]
    merged = pd.merge(local_df, gf[["bowl_style","global_SR"]], on="bowl_style", how="left")
    merged["ExpRuns"] = merged["Balls"] * (merged["global_SR"] / 100)
    merged["Impact/100b"] = merged.apply(
        lambda r: round(((r["Runs"] - 20 * r["Dismissals_count"]) / r["Balls"]) * 100, 2)
                  if r["Balls"] > 0 else 0, axis=1
    )
    bowlABB = {
        "LB": "Left Arm Fast",
        "LBG": "Left Arm Fast (Gun)",
        "LF": "Left Fast",
        "LFM": "Left Fast Medium",
        "LM": "Left Medium",
        "LMF": "Left Medium Fast",
        "LWS": "Left Wrist Spin",
        "OB": "Off Break",
        "RF": "Right Fast",
        "RFM": "Right Fast Medium",
        "RMF": "Right Medium Fast",
        "RM": "Right Medium",
        "RM/OB/LB": "Right Medium/Off Break/Left Arm",
        "SLA": "Slow Leg Spin"
    }
    merged["Bowl Style"] = merged["bowl_style"].apply(lambda x: bowlABB.get(x, x))
    merged.drop(columns=["bowl_style"], inplace=True)
    for c in ["SR", "Dismissals", "Boundary%", "Dot%", "Ave", "ExpRuns"]:
        merged[c] = merged[c].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) and math.isfinite(x) else "inf")
    merged["Balls"] = merged["Balls"].astype(int)
    merged["Runs"] = merged["Runs"].astype(int)
    merged = merged[[
        "Bowl Style", "Balls", "Runs", "SR", "Dismissals",
        "Boundary%", "Dot%", "Impact/100b"
    ]]
    return merged

def main():
    st.set_page_config(page_title="T20: SPORTS", layout="wide")
    st.title("T20 Cricket Sports\nData Science Lab Project by Sai Arun | Mustafa | Mitali")

    df = ld()
    df_global = df.copy()

    batters = sorted(df["bat"].dropna().unique())
    selected_batter = st.sidebar.selectbox("Select Batter", batters)
    if "year" in df.columns:
        minY = int(df["year"].min())
        maxY = int(df["year"].max())
        year_range = st.sidebar.slider("Year Range", minY, maxY, (minY, maxY))
    else:
        year_range = None
    bowler_type = st.sidebar.radio("Bowler Type", ["All", "Spin", "Pace"])

    sub = df[df["bat"] == selected_batter].copy()
    if year_range and "year" in sub.columns:
        sub = sub[(sub["year"] >= year_range[0]) & (sub["year"] <= year_range[1])]
    if bowler_type != "All":
        sub = sub[sub["bowler_type"] == bowler_type]

    if sub.empty:
        st.warning("No data found for the selected filters.")
        return

    sub = shift_coords(sub)
    sub = sd(sub)

    hand = "RHB/LHB?"
    if "bat_hand" in sub.columns and sub["bat_hand"].dropna().size > 0:
        hand = sub["bat_hand"].dropna().iloc[0]

    total_runs = sub["batruns"].sum()
    balls_faced = len(sub)
    outs = sub["dismissal"].notna().sum()
    sr_val = 100 * total_runs / balls_faced if balls_faced > 0 else 0
    bound_ct = sub[sub["batruns"].isin([4,6])].shape[0]
    bound_pct = 100 * bound_ct / balls_faced if balls_faced > 0 else 0
    dot_ct = sub[sub["batruns"] == 0].shape[0]
    dot_pct = 100 * dot_ct / balls_faced if balls_faced > 0 else 0
    ave_val = total_runs / outs if outs > 0 else float("inf")

    intro_tab, analysis_tab, help_tab = st.tabs(["Introduction", "Data Analysis", "Help"])

    with intro_tab:
        st.header("Introduction to Cricket, T20, and IPL")
        st.markdown("""
        ### Cricket: A Global Sport
        Cricket is one of the world's most popular sports, with a history dating back to the 16th century in England. It is a bat-and-ball game played between two teams of 11 players each, where the objective is to score runs by hitting a ball bowled at a wicket. The sport has various formats, including Test matches (up to five days), One Day Internationals (ODIs, 50 overs per side), and the fast-paced Twenty20 (T20), which has revolutionized modern cricket.

        Cricket enjoys a massive following, especially in countries like India, England, Australia, Pakistan, South Africa and Afghanistan. Governed by the International Cricket Council (ICC), it combines strategy, skill, and athleticism, making it a unique and enduring game.

        ### T20: The Game Changer
        Twenty20 (T20) cricket, introduced in 2003 by the England and Wales Cricket Board (ECB), is the shortest format of the sport. Each team faces a maximum of 20 overs (120 legal deliveries), leading to matches that typically last about three hours. This brevity, combined with aggressive batting, frequent boundaries, and high entertainment value, has made T20 a global phenomenon.

        Key features of T20 include:
        - **Fast Pace**: Limited overs encourage explosive play, with batters aiming for boundaries (4 runs) and sixes (6 runs).
        - **Bowling Variety**: Bowlers employ diverse strategies, from pace to spin, to restrict scoring.
        - **Fan Appeal**: Shorter duration and dynamic action attract a broader audience, including younger fans.

        The first T20 World Cup was held in 2007, won by India, cementing T20’s status as a transformative format. As of 2025, the ICC continues to expand T20 through international tournaments and domestic leagues.

        ### IPL: India’s Cricketing Extravaganza
        The Indian Premier League (IPL), launched in 2008 by the Board of Control for Cricket in India (BCCI), is the world’s premier T20 franchise league. Combining top international talent with Indian players, the IPL has become a cultural and commercial juggernaut.

        **Key Highlights**:
        - **Teams**: As of 2025, the IPL features 10 teams, including iconic franchises like Mumbai Indians, Chennai Super Kings, and Royal Challengers Bangalore.
        - **Format**: Played annually (typically March-May), it includes a group stage followed by playoffs, culminating in a grand final.
        - **Impact**: The IPL has redefined cricket economics, with player auctions, massive broadcasting deals, and sponsorships. In 2024, its brand value was estimated at over $10 billion.
        
        The IPL not only entertains millions but also serves as a data-rich platform for analysis, blending sport with cutting-edge analytics, as explored in this app.
        """)

    with analysis_tab:
        st.header("Data Analysis")
        st.markdown(f"""
        <div style='background-color:#1e1e1e; color:white; padding:20px; border-radius:5px; margin-bottom:20px;'>
          <h2>{selected_batter} | {hand}</h2>
          <p>
            Runs: {total_runs} | 
            Balls: {balls_faced} | 
            SR: {sr_val:.2f} | 
            Boundary%: {bound_pct:.2f}% | 
            Dot%: {dot_pct:.2f}% | 
            Ave: {"∞" if ave_val==float("inf") else f"{ave_val:.2f}"}
          </p>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Wagon Wheels", "Dismissal Analysis", "Bowling Style Stats",
            "Scoring Patterns Over Time"
        ])

        with tab1:
            st.subheader("Boundary Wagon Wheel: General vs. Intelligent")
            fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(14, 6))
            bdf = wagonGen(sub, ax_left)
            bdf = wagonSD(sub, ax_right)
            fig.tight_layout()
            st.pyplot(fig)

            if not bdf.empty:
                fours = len(bdf[bdf["batruns"] == 4])
                sixes = len(bdf[bdf["batruns"] == 6])
                total_boundaries = fours + sixes
                zone_counts = bdf["wagonZone"].value_counts()
                most_common_zone = zone_counts.idxmax() if not zone_counts.empty else "N/A"
                most_common_zone_count = zone_counts.max() if not zone_counts.empty else 0
                zone_percentage = (most_common_zone_count / total_boundaries * 100) if total_boundaries > 0 else 0
                off_side_zones = [1, 2, 3, 4]
                leg_side_zones = [5, 6, 7, 8]
                off_side_boundaries = len(bdf[bdf["wagonZone"].isin(off_side_zones)])
                leg_side_boundaries = len(bdf[bdf["wagonZone"].isin(leg_side_zones)])
                off_side_pct = (off_side_boundaries / total_boundaries * 100) if total_boundaries > 0 else 0
                leg_side_pct = (leg_side_boundaries / total_boundaries * 100) if total_boundaries > 0 else 0
                side_preference = "off-side" if off_side_boundaries > leg_side_boundaries else "leg-side"
                bdf["magnitude_raw"] = bdf["batruns"] * bdf["shot_difficulty"]
                zone_difficulty = bdf.groupby("wagonZone")["magnitude_raw"].mean().sort_values(ascending=False)
                hardest_zone = zone_difficulty.idxmax() if not zone_difficulty.empty else "N/A"
                hardest_zone_value = zone_difficulty.max() if not zone_difficulty.empty else 0

                st.markdown(f"""
                **Insights**:
                - **Boundary Distribution**: {selected_batter} hit {fours} fours and {sixes} sixes, totaling {total_boundaries} boundaries.
                - **Favorite Zone**: The most common zone for boundaries is Zone {most_common_zone}, accounting for {zone_percentage:.1f}% of boundaries.
                - **Side Preference**: {selected_batter} favors the {side_preference}, with {off_side_pct:.1f}% of boundaries on the off-side and {leg_side_pct:.1f}% on the leg-side.
                - **Shot Difficulty**: The intelligent wagon wheel highlights Zone {hardest_zone} as the area where {selected_batter} excels in difficult shots (average weighted score: {hardest_zone_value:.1f}).
                - **Takeaway**: {selected_batter} could focus on diversifying shot selection to reduce predictability, while bowlers might target less dominant zones to restrict boundary scoring.
                """)
            else:
                st.markdown("**Insights**: No boundaries recorded for this selection.")

            st.subheader("Wagon Zone Wheel")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            zone_df = wagonZoneWheel(sub, ax2)
            fig2.tight_layout()
            st.pyplot(fig2)

            if not zone_df.empty:
                top_zone_runs = zone_df.loc[zone_df["Runs"].idxmax()]
                top_zone_sr = zone_df.loc[zone_df["SR"].idxmax()]
                top_zone_runs_id = top_zone_runs["wagonZone"]
                top_zone_runs_value = top_zone_runs["Runs"]
                top_zone_runs_pct = top_zone_runs["pct_runs"]
                top_zone_sr_id = top_zone_sr["wagonZone"]
                top_zone_sr_value = top_zone_sr["SR"]
                weak_zone = zone_df.loc[zone_df["SR"].idxmin()]
                weak_zone_id = weak_zone["wagonZone"]
                weak_zone_sr = weak_zone["SR"]
                weak_zone_balls = weak_zone["Balls"]
                dismissals_per_zone = sub[sub["dismissal"].notna()].groupby("wagonZone").size().reset_index(name="Dismissals")
                zone_df = zone_df.merge(dismissals_per_zone, on="wagonZone", how="left").fillna({"Dismissals": 0})
                high_dismissal_zone = zone_df.loc[zone_df["Dismissals"].idxmax()] if not zone_df["Dismissals"].eq(0).all() else None
                high_dismissal_zone_id = high_dismissal_zone["wagonZone"] if high_dismissal_zone is not None else "N/A"
                high_dismissal_count = high_dismissal_zone["Dismissals"] if high_dismissal_zone is not None else 0

                st.markdown(f"""
                **Insights**:
                - **Top Scoring Zone**: Zone {top_zone_runs_id} is {selected_batter}'s most productive, contributing {top_zone_runs_value} runs ({top_zone_runs_pct:.1f}% of total runs).
                - **Highest Strike Rate**: Zone {top_zone_sr_id} has the highest strike rate at {top_zone_sr_value:.1f}, indicating aggressive scoring.
                - **Weak Zone**: Zone {weak_zone_id} is the least effective, with a strike rate of {weak_zone_sr:.1f} over {weak_zone_balls} balls.
                - **Dismissal Risk**: Zone {high_dismissal_zone_id} has the highest dismissal count ({high_dismissal_count} dismissals), making it a potential vulnerability.
                - **Takeaway**: {selected_batter} might target Zone {weak_zone_id} to improve scoring, while bowlers could exploit Zone {high_dismissal_zone_id} to increase dismissal chances.
                """)
            else:
                st.markdown("**Insights**: No valid zone data available for this selection.")

        with tab2:
            st.subheader(f"Percentage of Outs by Bowl Length for {selected_batter}")
            length_summary = (
                sub.assign(out_flag=sub['out'].astype(bool))
                .groupby('length')
                .agg(total_balls=('out_flag', 'size'), total_outs=('out_flag', 'sum'))
                .reset_index()
            )
            total_outs = length_summary['total_outs'].sum()
            if total_outs > 0:
                length_summary['out_percentage'] = 100 * length_summary['total_outs'] / total_outs
            else:
                length_summary['out_percentage'] = 0
            length_summary = length_summary.sort_values('out_percentage', ascending=False)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3 = sns.barplot(
                x='length', y='out_percentage', data=length_summary,
                order=length_summary['length'], palette='Set2'
            )
            for i, bar in enumerate(ax3.patches):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2, height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10
                )
            plt.title(f'Percentage of Outs by Bowl Length for {selected_batter}', fontsize=14, fontweight='bold')
            plt.xlabel('Bowl Length', fontsize=12)
            plt.ylabel('Percentage of Total Outs (%)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, length_summary['out_percentage'].max() + 15 if total_outs > 0 else 100)
            plt.tight_layout()
            st.pyplot(fig3)

            if total_outs > 0:
                max_length = length_summary.iloc[0]['length']
                max_pct = length_summary.iloc[0]['out_percentage']
                max_balls = length_summary.iloc[0]['total_balls']
                min_length = length_summary.iloc[-1]['length']
                min_pct = length_summary.iloc[-1]['out_percentage']
                min_balls = length_summary.iloc[-1]['total_balls']
                st.markdown(f"""
                **Insights**:
                - **Most Dangerous Length**: {max_length} deliveries dismiss {selected_batter} most often ({max_pct:.1f}% of outs, {int(max_balls)} balls faced), making it the best length to bowl.
                - **Safest Length**: {min_length} deliveries are safest for the batter, with only {min_pct:.1f}% of dismissals ({int(min_balls)} balls faced).
                - **Takeaway**: Bowlers should prioritize {max_length} deliveries to maximize dismissal chances, while {selected_batter} might focus on improving technique against this length.
                """)
            else:
                st.markdown(f"""
                **Insights**: No dismissals recorded for {selected_batter} under the current filters.  
                Try adjusting the filters to see dismissal patterns.
                """)

            st.subheader(f"Percentage of Outs by Bowl Line for {selected_batter}")
            if 'line' in sub.columns:
                sub['line'] = sub['line'].str.upper().str.replace(' ', '_')
                line_summary = sub[sub['out'] == 1]['line'].value_counts().reset_index()
                line_summary.columns = ['line', 'outs']
                total_outs = line_summary['outs'].sum()
                total_balls_per_line = sub.groupby('line').size().reset_index(name='total_balls')
                if total_outs > 0:
                    line_summary['percentage'] = 100 * line_summary['outs'] / total_outs
                else:
                    line_summary['percentage'] = 0
                line_summary = line_summary.sort_values('percentage', ascending=False)
                line_summary = line_summary.merge(total_balls_per_line, on='line', how='left').fillna({'total_balls': 0})
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                ax4 = sns.barplot(x='line', y='percentage', data=line_summary, order=line_summary['line'], palette='Set1')
                for p in ax4.patches:
                    ax4.text(p.get_x() + p.get_width() / 2, p.get_height(), f'{p.get_height():.1f}%', ha='center', va='bottom')
                plt.title(f'Percentage of Outs by Bowl Line for {selected_batter}')
                plt.xlabel('Bowl Line')
                plt.ylabel('Percentage of Outs (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig4)

                if total_outs > 0:
                    max_line = line_summary.iloc[0]['line']
                    max_pct = line_summary.iloc[0]['percentage']
                    max_balls = int(line_summary.iloc[0]['total_balls'])
                    min_line = line_summary.iloc[-1]['line']
                    min_pct = line_summary.iloc[-1]['percentage']
                    min_balls = int(line_summary.iloc[-1]['total_balls'])
                    st.markdown(f"""
                    **Insights**:
                    - **Most Dangerous Line**: {max_line} deliveries are most likely to dismiss {selected_batter} ({max_pct:.1f}% of outs, {max_balls} balls faced), making it the best line to target.
                    - **Safest Line**: {min_line} deliveries are least effective, with only {min_pct:.1f}% of dismissals ({min_balls} balls faced).
                    - **Takeaway**: Bowlers should focus on the {max_line} line to increase dismissal chances, while {selected_batter} might work on defending or attacking this line more effectively.
                    """)
                else:
                    st.markdown(f"""
                    **Insights**: No dismissals recorded for {selected_batter} under the current filters.  
                    Try adjusting the filters to see dismissal patterns.
                    """)
            else:
                st.markdown("**Error**: 'line' column not found in the dataset. This visualization cannot be generated.")

            st.subheader(f"Percentage of Outs by Bowl Line and Length for {selected_batter}")
            if 'line' in sub.columns and 'length' in sub.columns:
                sub['line'] = sub['line'].str.upper().str.replace(' ', '_')
                line_map = {'WIDE_DOWN_LEG': 0, 'DOWN_LEG': 1, 'ON_THE_STUMPS': 2, 'OUTSIDE_OFFSTUMP': 3, 'WIDE_OUTSIDE_OFFSTUMP': 4}
                length_map = {'YORKER': 0, 'FULL_TOSS': 1, 'FULL': 2, 'GOOD_LENGTH': 3, 'SHORT_OF_A_GOOD_LENGTH': 4, 'SHORT': 5}
                sub_filtered = sub[sub['line'].isin(line_map.keys()) & sub['length'].isin(length_map.keys())]
                if not sub_filtered.empty:
                    pitch_data = sub_filtered[sub_filtered['out'] == 1].groupby(['line', 'length']).size().reset_index(name='out_count')
                    total_balls_per_combination = sub_filtered.groupby(['line', 'length']).size().reset_index(name='total_balls')
                    pitch_data['line_idx'] = pitch_data['line'].map(line_map).astype(float)
                    pitch_data['length_idx'] = pitch_data['length'].map(length_map).astype(float)
                    pitch_data = pitch_data.dropna(subset=['line_idx', 'length_idx'])
                    total_outs = pitch_data['out_count'].sum()
                    if total_outs > 0:
                        pitch_data['out_percentage'] = (pitch_data['out_count'] / total_outs) * 100
                    else:
                        pitch_data['out_percentage'] = 0
                    all_lines = list(line_map.values())
                    all_lengths = list(length_map.values())
                    all_combinations = pd.DataFrame(
                        [(length_idx, line_idx) for length_idx in all_lengths for line_idx in all_lines],
                        columns=['length_idx', 'line_idx']
                    )
                    heatmap_data = all_combinations.merge(
                        pitch_data[['line_idx', 'length_idx', 'out_percentage', 'out_count']],
                        on=['length_idx', 'line_idx'],
                        how='left'
                    ).fillna({'out_percentage': 0, 'out_count': 0})
                    heatmap_data = heatmap_data.merge(
                        total_balls_per_combination,
                        left_on=['line_idx', 'length_idx'],
                        right_on=[total_balls_per_combination['line'].map(line_map), total_balls_per_combination['length'].map(length_map)],
                        how='left'
                    ).fillna({'total_balls': 0})
                    pivot_outs_percent = heatmap_data.pivot(index='length_idx', columns='line_idx', values='out_percentage')
                    fig5, ax5 = plt.subplots(figsize=(10, 12))
                    ax5.add_patch(Rectangle((-1, -1), 6, 1, fill=True, color='white', alpha=0.7))
                    ax5.add_patch(Rectangle((-1, 5.5), 6, 1, fill=True, color='white', alpha=0.7))
                    ax5.plot([2.5, 2.5], [-1, 6], color='black', linestyle='--')
                    if total_outs > 0:
                        sns.heatmap(pivot_outs_percent, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Outs (%)'},
                                    ax=ax5, linewidths=1, linecolor='gray', alpha=1.0)
                    else:
                        sns.heatmap(pivot_outs_percent, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Outs (%)'},
                                    ax=ax5, linewidths=1, linecolor='gray', alpha=1.0)
                        plt.text(2.5, 3, f'No dismissals for {selected_batter}', ha='center', va='center', fontsize=12, color='black')
                    plt.title(f'Percentage of Outs by Bowl Line and Length for {selected_batter}', fontsize=14, fontweight='bold', pad=20)
                    plt.xlabel('Bowl Line (Left to Right)', fontsize=12)
                    plt.ylabel('Bowl Length (Top to Bottom)', fontsize=12)
                    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['Wide Down Leg', 'Down Leg', 'On Stumps', 'Outside Off', 'Wide Outside Off'], rotation=45, ha='right')
                    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ['Yorker', 'Full Toss', 'Full', 'Good Length', 'Short of Good', 'Short'])
                    ax5.set_xlim(-0.3, 5)
                    ax5.set_ylim(-0.1, 6)
                    plt.subplots_adjust(top=0.9)
                    st.pyplot(fig5)

                    if total_outs > 0:
                        max_idx = heatmap_data['out_percentage'].idxmax()
                        max_line = heatmap_data.loc[max_idx, 'line']
                        max_length = heatmap_data.loc[max_idx, 'length']
                        max_pct = heatmap_data.loc[max_idx, 'out_percentage']
                        max_balls = int(heatmap_data.loc[max_idx, 'total_balls'])
                        non_zero = heatmap_data[heatmap_data['out_percentage'] > 0]
                        if not non_zero.empty:
                            min_idx = non_zero['out_percentage'].idxmin()
                            min_line = non_zero.loc[min_idx, 'line']
                            min_length = non_zero.loc[min_idx, 'length']
                            min_pct = non_zero.loc[min_idx, 'out_percentage']
                            min_balls = int(non_zero.loc[min_idx, 'total_balls'])
                            min_text = f"- **Safest Combination**: The {min_line} {min_length} combination is least effective, with only {min_pct:.1f}% of dismissals ({min_balls} balls faced)."
                        else:
                            min_text = "- **Safest Combination**: No other combinations resulted in dismissals."
                        st.markdown(f"""
                        **Insights**:
                        - **Most Dangerous Combination**: The {max_line} {max_length} combination is most effective, accounting for {max_pct:.1f}% of {selected_batter}'s dismissals ({max_balls} balls faced).
                        {min_text}
                        - **Takeaway**: Bowlers should target the {max_line} {max_length} combination to maximize dismissal chances, while {selected_batter} should be cautious against this combination.
                        """)
                    else:
                        st.markdown(f"""
                        **Insights**: No dismissals recorded for {selected_batter} under the current filters.  
                        Try adjusting the filters to see dismissal patterns.
                        """)
                else:
                    st.markdown("**Warning**: No valid data after filtering.")
            else:
                st.markdown("**Error**: 'line' or 'length' column missing.")

        with tab3:
            st.subheader("Specifications w.r.t Bowl Style")
            sty_table = tab(sub, df_global)
            if not sty_table.empty:
                st.dataframe(sty_table)
                max_sr_row = sty_table.loc[sty_table['SR'].idxmax()]
                max_sr_style = max_sr_row['Bowl Style']
                max_sr_value = max_sr_row['SR']
                max_sr_balls = max_sr_row['Balls']
                non_zero_runs = sty_table[sty_table['Runs'] > 0]
                if not non_zero_runs.empty:
                    min_sr_row = non_zero_runs.loc[non_zero_runs['SR'].idxmin()]
                    min_sr_style = min_sr_row['Bowl Style']
                    min_sr_value = min_sr_row['SR']
                    min_sr_balls = min_sr_row['Balls']
                    min_sr_text = f"- **Least Comfortable Style**: {selected_batter} struggles against {min_sr_style}, with a strike rate of {min_sr_value:.1f} over {min_sr_balls} balls."
                else:
                    min_sr_text = "- **Least Comfortable Style**: No bowling styles resulted in runs scored."
                max_dismiss_row = sty_table.loc[sty_table['Dismissals'].idxmax()]
                max_dismiss_style = max_dismiss_row['Bowl Style']
                max_dismiss_count = max_dismiss_row['Dismissals']
                max_dismiss_balls = max_dismiss_row['Balls']
                max_boundary_row = sty_table.loc[sty_table['Boundary%'].idxmax()]
                max_boundary_style = max_boundary_row['Bowl Style']
                max_boundary_pct = max_boundary_row['Boundary%']
                max_boundary_balls = max_boundary_row['Balls']
                max_dot_row = sty_table.loc[sty_table['Dot%'].idxmax()]
                max_dot_style = max_dot_row['Bowl Style']
                max_dot_pct = max_dot_row['Dot%']
                max_dot_balls = max_dot_row['Balls']
                max_impact_row = sty_table.loc[sty_table['Impact/100b'].idxmax()]
                max_impact_style = max_impact_row['Bowl Style']
                max_impact_value = max_impact_row['Impact/100b']
                max_impact_balls = max_impact_row['Balls']

                st.markdown(f"""
                **Insights**:
                - **Most Comfortable Style**: {selected_batter} scores freely against {max_sr_style}, with a strike rate of {max_sr_value:.1f} over {max_sr_balls} balls.
                {min_sr_text}
                - **Most Dangerous Style**: {max_dismiss_style} has dismissed {selected_batter} {max_dismiss_count} times over {max_dismiss_balls} balls, making it the most effective.
                - **Most Aggressive Against**: {selected_batter} hits boundaries most frequently against {max_boundary_style}, with a boundary percentage of {max_boundary_pct:.1f}% over {max_boundary_balls} balls.
                - **Most Restrictive Style**: {max_dot_style} restricts {selected_batter} the most, with a dot ball percentage of {max_dot_pct:.1f}% over {max_dot_balls} balls.
                - **Highest Impact Style**: {max_impact_style} has the highest impact against {selected_batter}, with an impact of {max_impact_value:.1f} per 100 balls over {max_impact_balls} balls.
                - **Takeaway**: {selected_batter} should work on improving scoring against {max_dot_style} to reduce dot balls and be cautious against {max_dismiss_style} to avoid dismissals. Bowlers should use {max_dismiss_style} to target {selected_batter}'s weaknesses.
                """)
            else:
                st.write("No bowling style data found.")

        with tab4:
            st.subheader(f"Scoring Patterns Over Time for {selected_batter}")
            if 'year' in sub.columns:
                yearly_stats = sub.groupby('year').agg(
                    total_runs=('batruns', 'sum'),
                    total_balls=('batruns', 'count'),
                    dismissals=('dismissal', lambda x: x.notna().sum()),
                    boundaries=('batruns', lambda x: x.isin([4, 6]).sum()),
                    dots=('batruns', lambda x: (x == 0).sum())
                ).reset_index()
                yearly_stats['strike_rate'] = 100 * yearly_stats['total_runs'] / yearly_stats['total_balls']
                years = sorted(yearly_stats['year'].unique())
                fig6, ax6 = plt.subplots(figsize=(10, 6))
                ax6.plot(yearly_stats['year'], yearly_stats['strike_rate'], marker='o', label='Strike Rate', color='blue')
                ax6.set_xlabel('Year')
                ax6.set_ylabel('Strike Rate', color='blue')
                ax6.tick_params(axis='y', labelcolor='blue')
                ax6_2 = ax6.twinx()
                ax6_2.plot(yearly_stats['year'], yearly_stats['total_runs'], marker='o', label='Total Runs', color='orange')
                ax6_2.set_ylabel('Total Runs', color='orange')
                ax6_2.tick_params(axis='y', labelcolor='orange')
                ax6.set_xticks(years)
                ax6.set_xticklabels(years, rotation=45)
                fig6.legend(loc='upper left')
                plt.title(f'Strike Rate and Runs Over Time for {selected_batter}')
                plt.tight_layout()
                st.pyplot(fig6)

                fig7, ax7 = plt.subplots(figsize=(8, 5))
                sns.barplot(x='year', y='dismissals', data=yearly_stats, palette='Set2')
                ax7.set_xticks(range(len(years)))
                ax7.set_xticklabels(years, rotation=45)
                plt.title(f'Dismissals Per Year for {selected_batter}')
                plt.xlabel('Year')
                plt.ylabel('Number of Dismissals')
                plt.tight_layout()
                st.pyplot(fig7)

                if not yearly_stats.empty:
                    peak_sr_year = yearly_stats.loc[yearly_stats['strike_rate'].idxmax(), 'year']
                    peak_sr_value = yearly_stats['strike_rate'].max()
                    peak_runs_year = yearly_stats.loc[yearly_stats['total_runs'].idxmax(), 'year']
                    peak_runs_value = yearly_stats['total_runs'].max()
                    max_dismissals_year = yearly_stats.loc[yearly_stats['dismissals'].idxmax(), 'year']
                    max_dismissals_count = yearly_stats['dismissals'].max()
                    st.markdown(f"""
                    **Insights**:
                    - **Peak Strike Rate**: {selected_batter} achieved their highest strike rate of {peak_sr_value:.1f} in {peak_sr_year}.
                    - **Peak Scoring Year**: The most runs ({peak_runs_value}) were scored in {peak_runs_year}.
                    - **Highest Dismissals**: {selected_batter} was dismissed {max_dismissals_count} times in {max_dismissals_year}, indicating a challenging year.
                    - **Takeaway**: {selected_batter} should aim to replicate their {peak_sr_year} form, while bowlers can exploit recent trends (e.g., higher dismissals in {max_dismissals_year}) to target weaknesses.
                    """)
            else:
                st.markdown("**Error**: 'year' column not found.")

    with help_tab:
        st.header("Help: Understanding the Variables")
        st.markdown("""
        This section explains the key variables used in the app, based on the dataset `IPL_2018_2024.xlsx`:

        - **bat**: Batter’s name.
        - **batruns**: Runs scored by the batter on a delivery (e.g., 0, 1, 2, 4, 6).
        - **out**: Indicates if the batter was out (1) or not (0).
        - **dismissal**: Type of dismissal (e.g., caught, bowled), if applicable.
        - **wagonX, wagonY**: Coordinates of where the ball landed or was hit, used for wagon wheel visualizations.
        - **wagonZone**: Zone (1-8) where the ball was hit, representing field areas (e.g., cover, mid-wicket).
        - **line**: Line of the delivery (e.g., "outside offstump", "down leg").
        - **length**: Length of the delivery (e.g., "full", "short").
        - **bowl_style**: Bowling style (e.g., "RF" for Right Fast, "OB" for Off Break).
        - **year**: Year of the match.
        - **bat_hand**: Batter’s handedness (e.g., "RHB" for right-handed batter).
        - **bowler_type**: Derived as "Spin", "Pace", or "Unknown" based on `bowl_style`.
        - **shot_difficulty**: Calculated metric reflecting how challenging a shot is based on line, length, and zone.

        These variables drive the visualizations and insights, helping analyze batter performance in the IPL context.
        """)

if __name__ == "__main__":
    main()
