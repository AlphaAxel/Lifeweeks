import io
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import streamlit as st

# ------------------ Mortality model (Gompertz–Makeham) ------------------
alpha = 0.0002893243687451036
beta  = 0.10425701482947486
c     = 0.0
anchor_age = 28.0
WEEKS_PER_ROW = 52

def hazard(age_years: np.ndarray) -> np.ndarray:
    return c + alpha * np.exp(beta * (age_years - anchor_age))

def survival_from_birth(age_years: np.ndarray) -> np.ndarray:
    return np.exp(- c*age_years - (alpha/beta)*np.exp(-beta*anchor_age)*(np.exp(beta*age_years) - 1.0))

# ------------------ Date helpers ------------------
def _naive_ymd(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day)

def col_for_date_within_year(date: datetime, weeks_per_row=52) -> int:
    d = _naive_ymd(date)
    jan1      = datetime(d.year, 1, 1)
    jan1_next = datetime(d.year + 1, 1, 1)
    frac      = (d - jan1).total_seconds() / (jan1_next - jan1).total_seconds()
    return int(np.floor(frac * weeks_per_row)) % weeks_per_row

def week_row_col_for_date(date: datetime, birth_date: datetime, weeks_per_row=52):
    d  = _naive_ymd(date)
    bd = _naive_ymd(birth_date)
    row = d.year - bd.year
    col = col_for_date_within_year(d, weeks_per_row)
    return row, col

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="LifeWeeks: Risk & Survival", page_icon="📅", layout="centered")
st.title("LifeWeeks: Risk & Survival")

dob = st.date_input("Date of birth", value=datetime(1997,5,25).date())

st.caption(
    "Each dot is a week of life. Colors show annual risk (log scale, jet). "
    "Greyscale shows conditional survival from today. "
    "Today is marked with a black ring."
)

# ------------------ Parameters ------------------
YEARS_TO_SHOW = 100
fade_base = 0.01
fade_gain = 0.45
dot_size  = 8

# ------------------ Build grid ------------------
birth_date = datetime(dob.year, dob.month, dob.day)
today = datetime.now(timezone.utc)

years = YEARS_TO_SHOW
cols  = WEEKS_PER_ROW

y_idx = np.arange(years)[:, None]
w_idx = np.arange(cols)[None, :]
ages  = y_idx + w_idx / WEEKS_PER_ROW

S_uncond  = survival_from_birth(ages)
MU_grid   = hazard(ages)

birth_col_0 = col_for_date_within_year(birth_date, WEEKS_PER_ROW)

today_naive = _naive_ymd(today)
birth_naive = _naive_ymd(birth_date)
a0 = max(0.0, (today_naive - birth_naive).days / 365.2425)

S0 = float(survival_from_birth(np.array([a0]))[0])
eps = 1e-12
S_cond = np.where(ages < a0, 1.0, S_uncond / max(S0, eps))

x_list, y_list, mu_list, s_list = [], [], [], []
row_mean_S = np.zeros(years)
for r in range(years):
    allowed = np.arange(birth_col_0, cols) if r == 0 else np.arange(cols)
    x_list.append(allowed)
    y_list.append(np.full(allowed.size, r))
    mu_list.append(MU_grid[r, allowed])
    s_list.append(S_cond[r, allowed])
    row_mean_S[r] = float(np.mean(S_cond[r, allowed]))

x = np.concatenate(x_list)
y = np.concatenate(y_list)
mu_vals = np.concatenate(mu_list)
s_vals  = np.concatenate(s_list)

present_row, present_col = week_row_col_for_date(today, birth_date, WEEKS_PER_ROW)
present_ok = (0 <= present_row < years and not (present_row == 0 and present_col < birth_col_0))

# ------------------ Plot ------------------
FIG_W, FIG_H = 8.27, 11.69
FONT_SIZE_AXIS = 7
YEAR_X = -1.5
MONTH_PAD = 10

month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_pos = (np.linspace(0, cols, 12, endpoint=False) + cols/12/2.0)
month_pos = np.clip(month_pos, 0, cols-1)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
plt.subplots_adjust(left=0.10, right=0.84, top=0.95, bottom=0.08)

norm_surv = Normalize(vmin=0.0, vmax=1.0)
ax.scatter(x, y, c=s_vals, cmap="Greys", norm=norm_surv,
           s=dot_size, marker="o", edgecolors="none", zorder=4, alpha=0.28)

norm_risk = LogNorm(vmin=max(mu_vals.min(), 1e-7), vmax=mu_vals.max())
cmap_risk = plt.get_cmap("jet")
risk_rgba = cmap_risk(norm_risk(mu_vals))
risk_rgba[:, 3] = np.clip(fade_base + fade_gain * s_vals, 0.0, 1.0)
ax.scatter(x, y, s=dot_size, marker="o", facecolors=risk_rgba, edgecolors="none", zorder=5)

ax.invert_yaxis()
for side in ["left","right","top","bottom"]:
    ax.spines[side].set_visible(False)
ax.grid(False)
ax.set_xlim(-1.8, cols - 0.5)
ax.set_ylim(years - 0.5, -0.5)
ax.tick_params(axis="both", which="both", length=0, labelleft=False, labelbottom=False)

cmap_grey = plt.cm.Greys
for r in range(years):
    ax.text(YEAR_X, r, str(birth_date.year + r), va="center", ha="right",
            fontsize=FONT_SIZE_AXIS, color=cmap_grey(row_mean_S[r]), zorder=6)

ax_top = ax.secondary_xaxis("top")
ax_top.set_xticks(month_pos)
ax_top.set_xticklabels(month_labels, fontsize=FONT_SIZE_AXIS)
ax_top.tick_params(axis="x", length=0, pad=MONTH_PAD)
for sp in ax_top.spines.values():
    sp.set_visible(False)

cax_risk = fig.add_axes([0.91, 0.20, 0.02, 0.58])
cbar_risk = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_risk, norm=norm_risk),
                         cax=cax_risk, orientation="vertical")
cbar_risk.set_label("Annual Risk", fontsize=8, labelpad=1)
cbar_risk.ax.yaxis.set_label_position('left')
cbar_risk.ax.yaxis.set_ticks_position('left')
cbar_risk.ax.tick_params(labelsize=FONT_SIZE_AXIS)
for sp in cbar_risk.ax.spines.values():
    sp.set_visible(False)

sm_surv = plt.cm.ScalarMappable(cmap="Greys", norm=norm_surv)
sm_surv.set_array([])
cax_surv = fig.add_axes([0.985, 0.20, 0.02, 0.58])
cbar_surv = fig.colorbar(sm_surv, cax=cax_surv, orientation="vertical")
cbar_surv.set_label("Survival", fontsize=8, labelpad=1)
cbar_surv.ax.yaxis.set_label_position('left')
cbar_surv.ax.yaxis.set_ticks_position('left')
cbar_surv.ax.tick_params(labelsize=FONT_SIZE_AXIS)
for sp in cbar_surv.ax.spines.values():
    sp.set_visible(False)

if present_ok:
    ax.scatter([present_col], [present_row], s=dot_size*5.0,
               facecolors="none", edgecolors="black", linewidths=0.6, zorder=40)

st.pyplot(fig, use_container_width=True)

# ------------------ Downloads ------------------
buf_png = io.BytesIO()
fig.savefig(buf_png, format="png", dpi=600)
buf_png.seek(0)

buf_pdf = io.BytesIO()
fig.savefig(buf_pdf, format="pdf")
buf_pdf.seek(0)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("⬇️ Download PNG (600 dpi)", data=buf_png,
                       file_name=f"life_weeks_jet_{birth_date.strftime('%Y%m%d')}.png",
                       mime="image/png")
with col_dl2:
    st.download_button("⬇️ Download PDF (vector)", data=buf_pdf,
                       file_name=f"life_weeks_jet_{birth_date.strftime('%Y%m%d')}.pdf",
                       mime="application/pdf")
