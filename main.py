from inference import process_video

video = 'first.mp4'

info_results = process_video(video, show_video=False, save_video=True, show_plate=True)
#df = pd.DataFrame.from_dict(info_results, orient='index').sort_index()