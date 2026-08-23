import yt_dlp

number_of_songs = 0

ydl_opts = { #setting tải xuống
    'format': 'bestaudio/best',
    'outtmpl': 'Dataset\Music\Real_therapy\%(title)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio', #sử dụng ffmpeg để trích xuất âm thanh từ video, chỉ giữ lại âm thanh
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],

    'sleep_interval': 5,
    'max_sleep_interval': 15,   # Thêm thời gian trừ để ko bị youtube block
}

ydl = yt_dlp.YoutubeDL(ydl_opts) # Tạo lệnh tại theo options đã thiết lập

## Tải nhạc Việt
# with open('music5.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         title = line.strip()
#         if title:   # Kiểm tra nếu dòng không rỗng
#             print("Downloading:", title)
#             ydl.download(f'ytsearch1:{title}')
#             number_of_songs += 1

# print(f"Total songs downloaded: {number_of_songs}")


# Tải nhạc therapy thực tế
ydl.download('ytsearch1: 30 minutes Autism, ADHD, SPD, and Aspergers Sensory Soothing Music')