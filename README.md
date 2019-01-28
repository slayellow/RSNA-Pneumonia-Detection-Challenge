RSNA-Pneumonia-Detection-Challenge
==================================

-	RSNA: The Radiological Society of North America(북미 방사선 협회)
-	Pneumonia: 폐렴
-	Site: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

Description
-----------

-	흉부 방사선 사진에서 폐의 opacity을 자동적으로 찾아야 한다.
-	일반적으로 폐렴 진단은 어려운 작업
	-	숙련된 전문가가 흉부 방사선 사진(CXR) 및 임상 병력, 생체 신호 및 실험실 검사를 통한 확인 필요
-	폐렴은 보통 CXR에서 opacity이 증가한 영역에서 나타남
	-	그러나 폐부종, 출혈, 부피감소, 폐암 또는 방사선 후 또는 외과적 변화와 같은 폐의 여러 가지 환경 때문에 CXR에 의한 폐렴 진단은 복잡함
	-	폐 밖에서도 늑막 공간의 체액은 CXR의 opacity가 증가한 것처럼 보이기 때문
-	CXR은 가장 일반적으로 수행되는 진단 이미지 연구

Evaluation
----------

-	IoU Threshold의 서로 다른 평균 정밀도로 평가
	-	예측한 bounding box와 ground truth의 IoU

![IoU](./Image/IoU.png)

Goal
----

-	알고리즘을 적용하여 나온 Detection의 Bounding box (x,y,width,height)을 실제 Ground Truth에 IoU를 적용하여 결과 평가

Submission File Format
----------------------

-	환자ID, PredictionString --> confidence, x, y, width height

![Submission_file](./Image/Submission_file.png)

Dataset
-------

-	Image: stage_2_train_images.zip / stage_2_test_images.zip
-	Label: stage_2_train_label.csv
-	Submission Sample: stage_2_sample_submission.csv
-	Detailed Information: stage_2_detailed_class_info.csv
	-	PatientIds / Bounding_box: x-min, y-min, width, height / Target: pneumonia or non-pneumonia
	-	Multiple row per PatientId
	-	Image: DICOM Format
	-	주어진 이미지에서 폐렴이 존재하는지 여부를 예측
	-	bounding box가 없으면 negative / bounding box가 존재하면 positive

![Dataset_Example](./Image/Dataset_Example.png)

### Data Exploring

-	stage_2_train_label.csv파일을 python을 통해 열면 다음과 같이 구성

![Dataset_1](./Image/Dataset_1.png)

![Dataset_2](./Image/Dataset_2.png)

-	patientId를 활용하여 이미지 파일을 연다.

	-	DICOM file(.dcm)으로 구성되어 있으며 python에선 pydicom을 통하여 열 수 있다.
	-	DICOM file 안의 Description은 아래와 같다.

![Dataset_3](./Image/Dataset_3.png)

-	이미지 파일은 Numpy형식으로 구성되며 1024x1024 grayscale로 이루어져 있다.

![Dataset_4](./Image/Dataset_4.png)

-	이미지 파일과 Label 정보를 Dictionary형태로 저장

![Dataset_4_1](./Image/Dataset_4_1.png)

-	이미지 파일에 Bounding box을 추가

![Dataset_5](./Image/Dataset_5.png)

-	데이터셋에는 폐렴은 아니지만 다른 증상이 있는 이미지들이 존재
-	stage_2_detailed_class_info.csv파일을 통해 확인 가능
-	Example

![Dataset_6_1](./Image/Dataset_6_1.png)

-	위 Label은 폐렴이 존재하지 않은 걸로 나온다.

![Dataset_6](./Image/Dataset_6.png)

-	이미지로 확인을 해보면 이상이 있는 것을 확인할 수 있다.

-	이미지에는 여러가지 이상이 존재하며, 이 결과들은 폐렴과 관련이 없다

	-	Ground Truth Label은 100% 객관적이며 알고리즘 개발시 이 점을 유의해서 개발 진행

Model
-----

Result
------

### Hyper Parameter

Discussion
----------
