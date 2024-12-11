import parselmouth

# Load the audio file
sound = parselmouth.Sound("s2_swwp2s.wav")

# Define the Praat script for phoneme segmentation
praat_script = """
Read from file... audio_sample.wav
To TextGrid (silences)... 0.5 0.5 0.1 no no no
"""

# Run the script
objects = parselmouth.praat.call(sound, "Run script...", praat_script)

# Save the resulting TextGrid
textgrid = objects[0]
textgrid.save("phoneme_segments.TextGrid", "text")

interval_tier = textgrid.get_tier_by_name("IntervalTier 1")

for interval in interval_tier.intervals:
    print(f"Phoneme: {interval.text}, Start: {interval.start_time}, End: {interval.end_time}")
