#include "SegDataset.h"
#include"utils/Augmentations.h"

std::vector<cv::Scalar> get_color_list(){
    std::vector<cv::Scalar> color_list = {
        cv::Scalar(0, 0, 0),
        cv::Scalar(128, 0, 0),
        cv::Scalar(0, 128, 0),
        cv::Scalar(128, 128, 0),
        cv::Scalar(0, 0, 128),
        cv::Scalar(128, 0, 128),
        cv::Scalar(0, 128, 128),
        cv::Scalar(128, 128, 128),
        cv::Scalar(64, 0, 0),
        cv::Scalar(192, 0, 0),
        cv::Scalar(64, 128, 0),
        cv::Scalar(192, 128, 0),
        cv::Scalar(64, 0, 128),
        cv::Scalar(192, 0, 128),
        cv::Scalar(64, 128, 128),
        cv::Scalar(192, 128, 128),
        cv::Scalar(0, 64, 0),
        cv::Scalar(128, 64, 0),
        cv::Scalar(0, 192, 0),
        cv::Scalar(128, 192, 0),
        cv::Scalar(0, 64, 128),
    };
    return color_list;
}


void show_mask(std::string json_path, std::string image_type) {
    using namespace std;
    using json = nlohmann::json;
    std::string image_path = replace_all_distinct(json_path, ".json", image_type);
    cv::Mat image = cv::imread(image_path);
    cv::Mat mask;
    mask.create(image.size(), CV_8UC3);

    std::ifstream jfile(json_path);
    json j;
    jfile >> j;
    size_t num_blobs = j["shapes"].size();

    for (int i = 0; i < num_blobs; i++)
    {
        std::string name = j["shapes"][i]["label"];
        size_t points_len = j["shapes"][i]["points"].size();
        cout << name << endl;
        std::vector<cv::Point> contour = {};
        for (int t = 0; t < points_len; t++)
        {
            int x = round(double(j["shapes"][i]["points"][t][0]));
            int y = round(double(j["shapes"][i]["points"][t][1]));
            cout << x << "\t" << y << endl;
            contour.push_back(cv::Point(x, y));
        }
        const cv::Point* ppt[1] = { contour.data() };
        int npt[] = { int(contour.size()) };
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255, 255, 255));
    }
    cv::imshow("mask", mask);
    cv::imshow("image", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void SegDataset::draw_mask(std::string json_path, cv::Mat &mask){
    std::ifstream jfile(json_path);
    nlohmann::json j;
    jfile >> j;
    size_t num_blobs = j["shapes"].size();


    for (int i = 0; i < num_blobs; i++)
    {
        std::string name = j["shapes"][i]["label"];
        size_t points_len = j["shapes"][i]["points"].size();
//        std::cout << name << std::endl;
        std::vector<cv::Point> contour = {};
        for (int t = 0; t < points_len; t++)
        {
            int x = round(double(j["shapes"][i]["points"][t][0]));
            int y = round(double(j["shapes"][i]["points"][t][1]));
//            std::cout << x << "\t" << y << std::endl;
            contour.push_back(cv::Point(x, y));
        }
        const cv::Point* ppt[1] = { contour.data() };
        int npt[] = { int(contour.size()) };
        cv::fillPoly(mask, ppt, npt, 1, name2color[name]);
    }
}

SegDataset::SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
                       std::vector<std::string> list_labels, std::vector<std::string> name_list,
					   trainTricks tricks, bool isTrain)
{
	this->tricks = tricks;
	this->name_list = name_list;
	this->resize_width = resize_width;
	this->resize_height = resize_height;
	this->list_images = list_images;
	this->list_labels = list_labels;
	this->isTrain = isTrain;
    for(int i=0; i<name_list.size(); i++){
        name2index.insert(std::pair<std::string, int>(name_list[i], i));
    }
    std::vector<cv::Scalar> color_list = get_color_list();
    if(name_list.size()>color_list.size()){
        std::cout<< "Num of classes exceeds defined color list, please add color to color list in SegDataset.cpp";
    }
    for(int i = 0; i<name_list.size(); i++){
        name2color.insert(std::pair<std::string, cv::Scalar>(name_list[i],color_list[i]));
    }
}

torch::data::Example<> SegDataset::get(size_t index) {
    std::string image_path = list_images.at(index);
    std::string label_path = list_labels.at(index);
    cv::Mat image = cv::imread(image_path);
    cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    draw_mask(label_path,mask);

    //Data augmentation like flip or rotate could be implemented here...
	auto m_data = Data(image, mask);
	if (isTrain) {
		m_data = Augmentations::Resize(m_data, resize_width, resize_height, 1);
		m_data = Augmentations::HorizontalFlip(m_data, tricks.horizontal_flip_prob);
		m_data = Augmentations::VerticalFlip(m_data, tricks.vertical_flip_prob);
		m_data = Augmentations::RandomScaleRotate(m_data, tricks.scale_rotate_prob, \
												  tricks.rotate_limit, tricks.scale_limit, \
												  tricks.interpolation, tricks.border_mode);
	}
	else {
		m_data = Augmentations::Resize(m_data, resize_width, resize_height, 1);
	}
    torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
    torch::Tensor colorful_label_tensor = torch::from_blob(m_data.mask.data, { m_data.mask.rows, m_data.mask.cols, 3 }, torch::kByte);
    torch::Tensor label_tensor = torch::zeros({ m_data.image.rows, m_data.image.cols});

    //encode "colorful" tensor to class_index meaning tensor, [w,h,3]->[w,h], pixel value is the index of a class
    for(int i = 0; i<name_list.size(); i++){
        cv::Scalar color = name2color[name_list[i]];
        torch::Tensor color_tensor = torch::tensor({color.val[0],color.val[1],color.val[2]});
        label_tensor = label_tensor + torch::all(colorful_label_tensor==color_tensor,-1)*i;
    }
    label_tensor = label_tensor.unsqueeze(0);
    return { img_tensor.clone(), label_tensor.clone() };
}
