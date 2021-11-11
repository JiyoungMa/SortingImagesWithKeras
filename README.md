# SortingImagesWithKeras

<h2> 3학년 2학기 데이터 사이언스 개론 강의 기말고사 대체 과제</h2>

- 음식 및 식당 사진에 대하여 '음식', '실내', '실외' 등으로 자동 분류하는 image classification 프로그램 개발 <br>
- CNN에 기반한 image classification 알고리즘 개발 및 성능 분석 <br>
- '음식', '실내', '실외'로 분류된 이미지 7만장를 학습 데이터로 사용 (Example Images 참고) <br>

<br>

<h3> 목차 </h3>

- [Training Model](#training-model)
- [Normalization](#normalization)
- [Layer](#layer)
- [Final Model](#final-model)

<br>

# Training Model
- train 및 text 데이터로 구분하여 모델을 학습시킴
- train 및 text 데이터로 이미지를 구분할 때, 데이터가 클래스별로 잘 섞일 수 있도록 seed를 123으로 설정

<br>

# Normalization
<h3> 외부 정규화 </h3>
- keras.preprocessing.image의 ImageDataGenerator를 사용하여 이미지의 각 픽셀 값을 0~1로 변환
- 위와 같은 외부 정규화 방법을 사용하였을 때, 가장 좋은 성능을 보였던 모델의 성능 평가
- <img src = "https://user-images.githubusercontent.com/50768959/141365179-794328b1-285d-4676-9c6c-ade1f2171525.png">
- <img src = "https://user-images.githubusercontent.com/50768959/141365248-b4bb741d-5677-4645-b84a-7f5b32006c0a.png">
- <img src = "https://user-images.githubusercontent.com/50768959/141365258-512db0ca-a0d2-47cf-acaf-0c0c1d80ac2d.png">

<h3> 모델 내 정규화 </h3>
- Exploding gradient 문제를 해결하기 위해 tensorflow.keras.layer의 Batch Normalization 사용
- Batch Normalization : 각 레이어에 들어가는 input을 정규화하여 학습을 가속하며, Vanishing / Exploding gradient 문제를 해결하기 위한 방법
- 위와 같은 레이어 구조에서 Batch Normalizastion을 사용한 모델의 성능 평가
- <img src = "https://user-images.githubusercontent.com/50768959/141365840-20f45216-dc90-44d3-8041-070d32fcb308.png">
- <img src = "https://user-images.githubusercontent.com/50768959/141365976-403331bb-8866-4cae-9f5e-9f5552e9d610.png">
- <img src = "https://user-images.githubusercontent.com/50768959/141365868-ba8930fd-5951-404a-a499-bebbbc4d933c.png">

<br>

# Layer
- Conv2D에서 kernel의 크기를 3으로 strides를 1로, MaxPooling2D에서 poolsize를 2로 하였을 때, 인풋 크기가 너무 작지 않을 정도로 레이어 선정 (인풋 크기가 너무 작을 때 정확도가 떨어질 것을 걱정)
- <img src = "https://user-images.githubusercontent.com/50768959/141366191-6fd3f3d0-2a94-4234-ba44-27525f97df16.png">

<br>

# Final Model
<h3> 최종 모델 성능 측정 </h3>
<img src = "https://user-images.githubusercontent.com/50768959/141366375-3e8d005b-5c30-4dc0-8634-d4e525a5adfc.png">
<img src = "https://user-images.githubusercontent.com/50768959/141366415-f48890c4-f2ab-4f49-a977-50fabac76f4e.png">
<img src = "https://user-images.githubusercontent.com/50768959/141366419-e278d18b-2b56-4cb3-8ad6-cf0a82fefa8b.png">

<h3> 최종 모델 Prediction Example </h3>
- 0 : Food, 1 : Interior, 2 : Exterior
<h5> Example 1 </h5>
<img src = "https://user-images.githubusercontent.com/50768959/141366678-05d72885-9939-47bb-8cda-360cc027efb8.png">
<img src = "https://user-images.githubusercontent.com/50768959/141366685-f24e3ee9-5efb-4cf5-a104-9aac7a77d401.png">

<h5> Example 2 </h5>
<img src = "https://user-images.githubusercontent.com/50768959/141366775-5ceb786b-c010-4f52-b951-084c9f37dc11.png">
<img src = "https://user-images.githubusercontent.com/50768959/141366786-50033b94-c756-45cb-8c35-e0726689837b.png">
