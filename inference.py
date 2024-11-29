from ultralytics import YOLO
import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from util import get_car, read_license_plate_UK, create_image, draw_border


plate_model = YOLO('./weights/best.pt')
plate_model.fuse()
car_model = YOLO('./weights/yolov8m.pt')
car_model.fuse()


def process_video(video, show_video=False, save_video=True, show_plate=True):

    video_path = os.path.join('videos', video)
    out_folder = os.path.join('results', os.path.splitext(video)[0])
    output_folder = os.path.join(out_folder, f'images_{os.path.splitext(video)[0]}')
    video_out_path= os.path.join(out_folder, f'out_{video}')
    csv_out_path = os.path.join(out_folder, f'results_{os.path.splitext(video)[0]}.csv')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+'/txt'):
        os.makedirs(output_folder+'/txt')
    if not os.path.exists(output_folder+'/bin'):
        os.makedirs(output_folder+'/bin')
    if not os.path.exists(output_folder+'/orig'):
        os.makedirs(output_folder+'/orig')

    info_results = {} 
    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Не удалось открыть видео: {video_path}")
    ret, frame = cap.read()

    if show_video:
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', frame.shape[1], frame.shape[0])
    
    if save_video:
        if video_out_path is None:
            raise ValueError('Output video path must be provided when save_video is True.')
        cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), 
                                  cap.get(cv2.CAP_PROP_FPS), 
                                  (frame.shape[1], frame.shape[0]))
        
    try:
        while ret:
            car_results = car_model.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml')[0]
            lst_car_results = []
            if car_results.boxes.id is not None:
                for result in car_results:
                    bbox = result.boxes.xyxy.cpu().numpy()
                    conf = float(result.boxes.conf.cpu().numpy().item())
                    car_id = int(result.boxes.id.cpu().numpy().item())
                    class_id = result.boxes.cls.cpu().numpy().astype(int)
                    if class_id in [2, 3, 5, 7]:
                        lst_car_results.append([bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], car_id, conf])
                    
            plate_results = plate_model.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml')[0]
            if plate_results.boxes.id is not None:
                for plate_result in plate_results:   
                    plate_bbox = plate_result.boxes.xyxy.cpu().numpy()
                    plate_conf = float(plate_result.boxes.conf.cpu().numpy().item())
                    plate_x1, plate_y1, plate_x2, plate_y2 = plate_bbox[0]
                    car_x1, car_y1, car_x2, car_y2, car_id, car_conf = get_car(plate_bbox[0], lst_car_results)

                    if car_id != -1:
                        plate_img = frame[int(plate_y1-10):int(plate_y2-10), int(plate_x1+10):int(plate_x2+10)]
                        plate_height, plate_width, _ = plate_img.shape
                        gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                        _, binary_plate_img = cv2.threshold(gray_plate_img, 62, 255, cv2.THRESH_BINARY_INV)
                        plate_txt, plate_txt_conf = read_license_plate_UK(binary_plate_img)

                        if car_id not in info_results:
                            orig_plate_img_filename = os.path.join(output_folder, 'orig', f'car_{car_id}_orig_plate.jpg')
                            cv2.imwrite(orig_plate_img_filename, plate_img)
                            info_results[car_id] = {
                                    'car_bbox': [int(car_x1), int(car_y1), int(car_x2), int(car_y2)],
                                    'car_conf': car_conf,
                                    'plate_bbox': [int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)],
                                    'plate_conf': plate_conf}

                        if plate_txt is not None and (car_id not in info_results or plate_txt_conf > info_results[car_id].get('plate_text_conf', 0)):
                            txt_plate_img = np.array(create_image(plate_txt))
                            txt_plate_img_filename = os.path.join(output_folder, 'txt', f'car_{car_id}_txt_plate.jpg')
                            bin_plate_img_filename = os.path.join(output_folder, 'bin', f'car_{car_id}_bin_plate.jpg')
                            orig_plate_img_filename = os.path.join(output_folder, 'orig', f'car_{car_id}_orig_plate.jpg')

                            cv2.imwrite(orig_plate_img_filename, plate_img)
                            cv2.imwrite(bin_plate_img_filename, binary_plate_img)
                            cv2.imwrite(txt_plate_img_filename, txt_plate_img)

                            info_results[car_id] = {
                                    'car_bbox': [int(car_x1), int(car_y1), int(car_x2), int(car_y2)],
                                    'car_conf': car_conf,
                                    'plate_bbox': [int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)],
                                    'plate_conf': plate_conf,
                                    'plate_text': plate_txt,
                                    'plate_text_conf': plate_txt_conf}

                        if car_id in info_results:
                            plate_img = np.array(Image.open(os.path.join(output_folder, 'orig', f'car_{car_id}_orig_plate.jpg')))
                            plate_height, plate_width, _ = plate_img.shape
                            plate_height = plate_height*2
                            plate_width = plate_width*2
                            y_pos_img = int(car_y1) - plate_height - 10
                            if y_pos_img >= 0: 
                                x_pos_img = int((car_x1 + car_x2 - plate_width) // 2) 
                                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR) 
                                plate_img = cv2.resize(plate_img, (plate_width, plate_height))          
                                frame[y_pos_img:y_pos_img + plate_height, x_pos_img:x_pos_img + plate_width] = plate_img 

                        cv2.putText(frame, f"Car {str(car_id)}", (int(car_x1+10), int(car_y1+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=4)
                        draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 10, line_length_x=25, line_length_y=25)
                        draw_border(frame, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 10, line_length_x=10, line_length_y=10)
            
            if show_video:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video:
                cap_out.write(frame)

            ret, frame = cap.read()

    finally:
        cap.release()
        if save_video:
            cap_out.release()
        cv2.destroyAllWindows()

        df = pd.DataFrame.from_dict(info_results, orient='index').sort_index()
        df.to_csv(csv_out_path, index_label='car_id', sep=';')

    return info_results