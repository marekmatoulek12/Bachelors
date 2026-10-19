import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def pacejka_force(x, b, c, d, e):
    shaped_slip = b * x
    return d * np.sin(
        c * np.arctan(shaped_slip - e * (shaped_slip - np.arctan(shaped_slip)))
    )


# Vystupni rozsah x (uhel smyku v rad / skluz)
x = np.linspace(0, 1.2, 500)

# Pocatecni parametry (uzij si ladeni)
b_init, c_init, d_init, e_init = 2.15, 1.3, 4.86, -1.6

fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(bottom=0.35)

(line,) = ax.plot(
    x,
    pacejka_force(x, b_init, c_init, d_init, e_init),
    linewidth=2.5,
    color="blue",
    label="Pacejka Krivka",
)
ax.set_title("Interaktivní hra s parametry Pacejky (Magic Formula)")
ax.set_xlabel("Skluz / Úhel smyku [rad / -]")
ax.set_ylabel("Sila F [N]")
ax.grid(True)
ax.set_ylim(-0.5, 7.0)

# Nastaveni pozic pro slidery
ax_b = plt.axes([0.15, 0.22, 0.7, 0.03])
ax_c = plt.axes([0.15, 0.17, 0.7, 0.03])
ax_d = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_e = plt.axes([0.15, 0.07, 0.7, 0.03])

s_b = Slider(
    ax_b, "B (Stiffness)", 0, 20.0, valinit=b_init, valfmt="%1.2f"
)  # B: tuhost / sklon v pocatku
s_c = Slider(
    ax_c, "C (Shape)", 0, 2.5, valinit=c_init, valfmt="%1.2f"
)  # C: tvarovy faktor
s_d = Slider(
    ax_d, "D (Peak)", 0.0, 10.0, valinit=d_init, valfmt="%1.2f"
)  # D: maxima sily (peak force)
s_e = Slider(
    ax_e, "E (Curvature)", -6.0, 1.0, valinit=e_init, valfmt="%1.2f"
)  # E: zakriveni spicky


def update(val):
    b = s_b.val
    c = s_c.val
    d = s_d.val
    e = s_e.val
    line.set_ydata(pacejka_force(x, b, c, d, e))
    fig.canvas.draw_idle()


s_b.on_changed(update)
s_c.on_changed(update)
s_d.on_changed(update)
s_e.on_changed(update)

plt.show()