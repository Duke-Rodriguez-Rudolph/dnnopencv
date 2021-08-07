#include<iostream>
#include<opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

int main() {
	//����ļ�
	string classFiles = "coco.names";
	//��ȡ����ļ�
	ifstream ifs(classFiles);
	//�洢���ı���
	vector<string> classes;
	string classLine;
	while (getline(ifs, classLine)) {
		classes.push_back(classLine);
	}

	//��ȡȨ���ļ���cfg�ļ�
	string modelWeights = "yolov3.weights";
	string modelCfg = "yolov3.cfg";
	dnn::Net net = dnn::readNetFromDarknet(modelCfg, modelWeights);

	//��ȡ�㼶��Ϣ
	vector<string> layerNames=net.getLayerNames();
	int lastLayerId = net.getLayerId(layerNames[layerNames.size() - 1]);
	Ptr<dnn::Layer> lastLayer = net.getLayer(dnn::DictValue(lastLayerId));


	//��ȡ�������Ϣ
	vector<string> outPutNames;
	vector<int> outLayers = net.getUnconnectedOutLayers();
	for (int i = 0; i < outLayers.size(); i++) {
		outPutNames.push_back(layerNames[outLayers[i] - 1]);
	}


	//��ȡ��Ƭ����Ƭ����
	Mat picture = imread("1.jpg");
	Mat blob=dnn::blobFromImage(picture,1.0f / 255,Size(320, 320),Scalar(0, 0, 0),true,false);
	net.setInput(blob);
	vector<Mat> probs;//�洢�����
	net.forward(probs, outPutNames);//ǰ�򴫲�


	//����ʱ��չʾ
	vector<double> layersTime;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTime) / freq;
	string label = format("Inference time: %.2f ms", t);
	putText(picture,label,Point(0, 15),FONT_HERSHEY_SIMPLEX,0.5,Scalar(255, 0, 0));



	//���ʶ��
	vector<Rect> boxes;
	vector<float> confidences; //�洢���Ŷ�
	vector<int> indices;  //����Ǽ�ֵ������Ϣ
	vector<string> labels;  //�����ǩ
	for (int i = 0; i < probs.size(); i++) {
		for (int j = 0; j < probs[i].rows; j++) {
			//����������ȡ����
			Mat scores = probs[i].row(j).colRange(5, probs[i].cols);
			Point classIdPoint;
			double confidence;//���Ŷ�
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.7) {
				int centerX = (int)(probs.at(i).at<float>(j, 0) * picture.cols);
				int centerY = (int)(probs.at(i).at<float>(j, 1) * picture.rows);
				int width = (int)(probs.at(i).at<float>(j, 2) * picture.cols);
				int height = (int)(probs.at(i).at<float>(j, 3) * picture.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;
				Rect objectRect(left, top, width, height);
				boxes.push_back(objectRect);
				confidences.push_back((float)confidence);
				String label = format("%s:%.4f",classes[classIdPoint.x].data(),confidence);
				labels.push_back(label);
			}
		}
	}

	//�Ǽ�ֵ����
	dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);


	//�����д���
	for (int i = 0; i < indices.size(); i++) {

		int index = indices[i];
		rectangle(picture, boxes[index], Scalar(255, 0, 0), 2);
		putText(picture, labels[index], Point(boxes[index].tl().x, boxes[index].tl().y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255),2);

	}
	imshow("img", picture);
	waitKey(0);
	return 0;
}