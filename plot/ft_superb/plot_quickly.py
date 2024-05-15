import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

png_folder = pathlib.Path("png")
csv_folder = pathlib.Path("csv")

png_folder.mkdir(exist_ok=True)

title_map = {
    'ic': 'intent classification',
    'er': 'emotion recognition',
    'pr': 'phoneme recognition',
    'sid': 'speaker identification',
    'vc': 'voice conversion',
    'asr-ood-zh': "mandarin speech recognition",
}
label_map = {
    'acc': 'accuracy',
    'mcd': 'melcepstrum distortion',
    'ar': 'speaker recognition acceptance rate',
    'per': 'phone error rate',
    'wer': 'word error rate',
    'cer': 'character error rate'
}

sns.set_style("darkgrid")

for csv_file in csv_folder.glob("*.csv"):
    df = pd.read_csv(csv_file)
    print(df.to_string())

    fig = sns.lineplot(df, x='hours-seen', y='metric-value', hue='batch-size', marker='o')
    fig.set_title(title_map[df['superb_task'][0]])

    plt.ylabel(label_map[df['metric'][0]])
    plt.xlabel("hours seen")
    plt.xticks([tick for tick in plt.xticks()[0]],
               [f'{tick / 1000:.0f}k' for tick in plt.xticks()[0]])

    # Set x-axis to logarithmic scale
    plt.xscale('log')

    # Set the limits of the x-axis
    plt.xlim(1000, 700000)

    plt.savefig(png_folder / f"{csv_file.stem}.png")
    plt.clf()