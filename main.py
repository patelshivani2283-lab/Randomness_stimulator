import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_growing_histogram(data, save_path="reports/plots/growing_hist.gif"):
    fig, ax = plt.subplots()

    # Create initial empty histogram
    n, bins, patches = ax.hist([], bins=20, range=[min(data), max(data)])
    ax.set_title("Growing Histogram Animation")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    # Animation update function
    def update(frame):
        ax.clear()
        ax.hist(data[:frame], bins=20, range=[min(data), max(data)], color='blue')
        ax.set_title(f"Growing Histogram (Step: {frame}/{len(data)})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    ani = animation.FuncAnimation(fig, update, frames=len(data), interval=50)

    # Save animation to GIF
    ani.save(save_path, writer='pillow')
    plt.close()

    print(f"Growing histogram saved at: {save_path}")
if __name__ == "__main__":
    # Example: generate random data
    data = np.random.normal(5, 10, 300)

    animate_growing_histogram(data)    