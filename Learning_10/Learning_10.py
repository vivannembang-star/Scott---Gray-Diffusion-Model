import numpy as np
import matplotlib.pyplot as plt



size = 100
rng = np.random.default_rng(3)

A = np.ones((size, size))
B = np.zeros((size, size))


for _ in range(12):
    i, j = rng.integers(5, size - 10, size = 2)
    B[i:i + 5,j:j + 5] = 1

B += 0.001 * rng.random(B.shape)

def laplcation(grid):
    return(
        np.roll(grid, 1, axis = 0)
        + np.roll(grid, -1, axis=0)
        + np.roll(grid, 1, axis=1)
        + np.roll(grid, -1, axis=1)
        - 4 * grid
    )

D_A = 1.0
D_B = 0.5
dt = 0.1

F = 0.0290
k = 0.0570

steps = 20000

max_rates = []


for step in range(1, steps + 1):
    diff_A = laplcation(A)
    diff_B = laplcation(B)
 
    change_A = (
        D_A * diff_A
        - A * B**2
        + F * (1 - A)
    )
 
    change_B = (
        D_B * diff_B
        + A * B**2
        - (F + k) * B
    )
 
    max_rate = max(
        np.max(np.abs(change_A)),
        np.max(np.abs(change_B))
    )
 
    A_new = A + dt * change_A
    B_new = B + dt * change_B
 
    if not (
        np.isfinite(A_new).all()
        and np.isfinite(B_new).all()
    ):
        raise ValueError(
            f"Simulation became unstable at step {step}. Try a smaller dt."
        )
 
    A = A_new
    B = B_new
 
    max_rates.append(max_rate)


print("Finished", steps, "steps")
print("Largest B value:", B.max())
print("Final maximum rate of change:", max_rates[-1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
axes[0].imshow(
    B,
    cmap="viridis",
    vmin=0,
    vmax=0.4
)
axes[0].set_title(f"B concentration - step {steps}")
axes[0].set_xticks([])
axes[0].set_yticks([])
 
axes[1].plot(range(1, steps + 1), max_rates)
axes[1].set_title("Maximum rate of change")
axes[1].set_xlabel("Simulation step")
axes[1].set_ylabel("Maximum absolute rate")
axes[1].set_yscale("log")
 
plt.tight_layout()

fig.savefig("diffusion_final.png", dpi=100)
print("Saved diffusion_final.png in the folder you ran this from")
plt.show()