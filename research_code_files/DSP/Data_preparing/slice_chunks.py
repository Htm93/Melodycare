import os
import librosa
import DSP_pipeline.config as config
import soundfile as sf

folder_dir = r"Dataset\Music\Real_therapy"
output_dir = r"Dataset\Music\Real_therapy\chunks"

#Create output file
os.makedirs(output_dir, exist_ok = True)

#Target sr used for audioldm2
TARGET_SR = config.SAMPLE_RATE

all_file = [f for f in os.listdir(folder_dir) if f.endswith(".wav")]

print(f"Starting batch process for {len(all_file)}")

def slice_chunk(input_dir, folder):
    for index, file_name in enumerate(folder):
        file_path = os.path.join(input_dir, file_name)        

        if os.path.exists(file_path):
            target_file = file_path
        else:
            print(f"[ERROR]: {file_name} not found")
            continue

        #Update working status
        print(f"[{index + 1}/{len(folder)}] Processing: {file_name}")

        #Load the file
        wav, sr = librosa.load(target_file, sr = TARGET_SR)

        #separate chunks
        sample_per_chunk = sr * 30
        total_chunks = len(wav) // sample_per_chunk

        #Slice the saved chunks
        #Loop for every number of chunk in the total chunks
        for chunk_idx in range(total_chunks):
            #Starting point of that chunk, as librosa work with sample rate => multiply the index with number of sample in 30s
            #Ex: 0 * sample_per_chunk = 0 mean starting from the begining
            #    1 * sample_per_chunk = sample_per_chunk mean starting right after the first one end
            start_sample = chunk_idx * sample_per_chunk
            end_sample = start_sample + sample_per_chunk #Add exact number of sample in 30s from the starting point

            #Slicing using numpy basic function
            audio_chunk = wav[start_sample : end_sample]

            #Create and save output file
            result_name = f"{file_name}_chunk_{chunk_idx+1:03d}.wav"
            result_file_path = os.path.join(output_dir, result_name)

            sf.write(result_file_path, audio_chunk, sr)

    print("-----Batch complete-----")

slice_chunk(folder_dir, all_file)