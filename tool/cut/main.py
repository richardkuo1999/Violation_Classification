import cv2
import json
from pathlib import Path


def cut_image(img, object_data, path):
    crop_img = img[int(object_data[1]):int(object_data[3]),
                    int(object_data[0]):int(object_data[2])]
    cv2.imwrite(str(path),crop_img)
    
def xywhn2xyxy(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y=x.copy()
    y[0] = w * (x[0] - x[2] / 2)  # top left x
    y[0] = 0 if y[0] < 0 else y[0]

    y[1] = h * (x[1] - x[3] / 2)  # top left y
    y[1] = 0 if y[1] < 0 else y[1]
    
    y[2] = w * (x[0] + x[2] / 2)  # bottom right x
    y[2] = w if y[2] > w else y[2]

    y[3] = h * (x[1] + x[3] / 2)  # bottom right y
    y[3] = h if y[3] > h else y[3]

    return y

obj_format = {
   'name':'',
   'category':'',
   'lat':0.0,
   'lon':0.0,
   "box2dxyxy": {
      "x1": 0,
      "y1": 0,
      "x2": 0,
      "y2": 0
  },
   "box2dxywh": {
      "x": 0,
      "y": 0,
      "w": 0,
      "h": 0
  }
}

if __name__ == "__main__":
    category_name = ['violations', 'legitimate']
    source = Path('input')
    save = Path('result')

    for gps_file in (source/'GPS').glob('*.txt'):
        video_TAG = gps_file.stem

        laneline_save = save / 'laneline'
        DriveArea_save = save / 'DriveArea'
        class_save = save / 'object'
        img_save = save / 'img'
        laneline_save.mkdir(parents=True, exist_ok=True)
        DriveArea_save.mkdir(parents=True, exist_ok=True)
        class_save.mkdir(parents=True, exist_ok=True)
        img_save.mkdir(parents=True, exist_ok=True)

        with open(gps_file) as fps_r:
          for data_id, gps_data in enumerate(fps_r.readlines(),1):
            txt_path = source / 'obj' / f'{video_TAG}_{data_id}.txt'
            name = txt_path.stem
            print(name)


            DriveArea_img = cv2.imread(str(source/'DriveArea' / f'{name}.png'))
            laneline_img = cv2.imread(str(source/'laneline' / f'{name}.png'))

            if (source / 'img' / f'{name}.jpg').exists():
              img = cv2.imread(str(source / 'img' / f'{name}.jpg'))
            elif (source / 'img' / f'{name}.png').exists():
              img = cv2.imread(str(source / 'img' / f'{name}.png'))
            else:
              print(f"{str(source / 'img' / f'{name}')} not exists")
              raise


            h,w = DriveArea_img.shape[:2]
            if txt_path.exists():
              with open(txt_path, 'r') as f:
                count = 0
                for line in f.readlines():
                  laneline = laneline_save / f'{name}_{count}.jpg'
                  DriveArea = DriveArea_save / f'{name}_{count}.jpg'
                  objsave = class_save / f'{name}_{count}.json'
                  imgsave = img_save / f'{name}_{count}.jpg'

                  object_data = line.strip('\n').split(' ')
                  category, object_data = int(object_data[0]),[float(i) for i in object_data[1:]]
                  object_data[2],object_data[3] = object_data[2]*2,object_data[3]*2
                  object_xyxy = xywhn2xyxy(object_data,w,h)
                  
                  cut_image(img, object_xyxy, imgsave)
                  cut_image(DriveArea_img, object_xyxy, DriveArea)
                  cut_image(laneline_img, object_xyxy, laneline)

                  obj_data = obj_format.copy()
                  obj_data['name'] = f'{name}_{count}'
                  obj_data['category'] = category_name[category]
                  obj_data['lat'], obj_data['lon'] = map(float, gps_data.strip('\n').split(','))
                  obj_data['box2dxyxy']['x1'],obj_data['box2dxyxy']['y1'],\
                      obj_data['box2dxyxy']['x2'],obj_data['box2dxyxy']['y2'] = object_xyxy
                  obj_data['box2dxywh']['x'],obj_data['box2dxywh']['y'],\
                      obj_data['box2dxywh']['w'],obj_data['box2dxywh']['h'] = object_data

                  obj_data = json.dumps(obj_data, indent=4)
                  with open(objsave, 'w') as obj_w:
                    obj_w.write(obj_data)

                  count+=1
            else:
                print("not exists")