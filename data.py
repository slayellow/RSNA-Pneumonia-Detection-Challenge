import glob, pylab, pandas as pd
import pydicom, numpy as np
import matplotlib.pyplot as plt

def parse_data(df, df_detailed):
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': 'E:/Data/RSNA Pneumonia Detection Challenge Data/stage_2_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': [],
                'class': ''}
        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    for n, row in df_detailed.iterrows():
        pid = row['patientId']
        if pid in parsed:
            parsed[pid]['class'] = row['class']

    return parsed

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]

    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

df = pd.read_csv('E:/Data/RSNA Pneumonia Detection Challenge Data/stage_2_train_labels.csv')
df_detailed = pd.read_csv('E:/Data/RSNA Pneumonia Detection Challenge Data/stage_2_detailed_class_info.csv')
parsed = parse_data(df, df_detailed)

summary = {}    # 이미지 Class별 종류
for n, row in df_detailed.iterrows():
    if row['class'] not in summary:
        summary[row['class']] = 0
    summary[row['class']] += 1


'''
print(df.iloc[0]) # Target:0 Image
print(df.iloc[4]) # Target:1 Image
patientId = df['patientId'][0]  # 0번째 patientId값
dcm_file = 'E:/Data/RSNA Pneumonia Detection Challenge Data/stage_2_train_images/%s.dcm' % patientId
dcm_data = pydicom.read_file(dcm_file)  # 이미지 파일 읽기
print(dcm_data) # 이미지 Description
im = dcm_data.pixel_array   # 이미지 픽셀 값
print(type(im)) # Type: Numpy array
print(im.dtype) # Data Type: Uint8
print(im.shape) # Data Shape: 1024x1024
pylab.imshow(im, cmap=pylab.cm.gist_gray)   # 이미지를 grayscale로 보여주기
pylab.axis('off')
pylab.show()    
print(parsed['00436515-870c-4b36-a041-de91049b9ab4'])   # 이미지 정보 저장
draw(parsed['00436515-870c-4b36-a041-de91049b9ab4'])    # 이미지 그리기
pylab.show()
print(df_detailed.iloc[0])  # Image Class
patientId = df_detailed['patientId'][0]
draw(parsed[patientId])
pylab.show()
print(summary) # 이미지 별 Class 종류 출력
'''
