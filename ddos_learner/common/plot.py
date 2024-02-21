
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shutil import rmtree


def regenerate_plots():    
    for subpath in os.listdir('stats'):
        for timestamp in os.listdir(f'stats/{subpath}'):            
            print(f"{subpath}/{timestamp}")
            df = pd.read_csv(f'stats/{subpath}/{timestamp}')
            timestamp=int(timestamp.replace('.csv', ''))
            if os.path.exists(f'plots/{subpath}/{timestamp}'):
                rmtree(f'plots/{subpath}/{timestamp}')            
            plot_stats(f"plots/{subpath}", **df.to_dict('list'))


def plot_stats(plot_path, idx, timestamp, acc, prec_attack, recl_attack, prec_normal, recl_normal, f1, lb, timest, qnt):    
    if isinstance(timestamp, list):
        timestamp=timestamp[0]
    plot_path=f'{plot_path}/{timestamp}'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)    
    
    for (name, data) in zip(['Accuracy', 'Precision', 'Recall' , 'Precision(Normal)', 'Recall(Normal)', 'F1-Score'], [acc, prec_attack, recl_attack, prec_normal, recl_normal, f1]):
        fig, ax = plt.subplots(figsize=(5, 5))
        
        fig.tight_layout(pad=2.5)
        ax.plot(idx, data, color='blue', marker='o', linewidth=1, markersize=1, label=f"{name} on next records")        
        ax.plot(idx, lb, color='red', marker='o', linewidth=.5, ms=.5, label='Ground truth')
        ax.set_ylim(0, 1)        
        # ax.set_title(name)
        ax.set_xlabel('# of records')
        ax.set_ylabel("Value")
        ax.legend(loc='lower left', bbox_to_anchor=(0, 0))
        
        fig.savefig(f'{plot_path}/{name}.png', dpi=500)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.tight_layout()

    ax.plot(timest, qnt)    
    ax.set_title("Flow")
    
    fig.savefig(f'{plot_path}/Flow.png', dpi=500)
    plt.close(fig)