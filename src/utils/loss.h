#pragma once
#include<torch/torch.h>

//prediction [NCHW], a tensor after softmax activation at C dim
//target [N1HW], a tensor refer to label
//num_class: int, equal to C, refer to class numbers, including background
torch::Tensor DiceLoss(torch::Tensor prediction, torch::Tensor target, int num_class) {
	auto target_onehot = torch::zeros_like(prediction); // N x C x H x W
	target_onehot.scatter_(1, target, 1);

	auto prediction_roi = prediction.slice(1, 1, num_class, 1);
	auto target_roi = target_onehot.slice(1, 1, num_class, 1);
	auto intersection = (prediction_roi*target_roi).sum();
	auto union_ = prediction_roi.sum() + target_roi.sum() - intersection;
	auto dice = (intersection + 0.0001) / (union_ + 0.0001);
	//cout << "prediction_roi: " << prediction_roi.sizes() << "\t" << "target roi: " << target_roi.sizes() << endl;
	//cout << "intersection: " << intersection << "\t" << "union: " << union_ << endl;
	//target_onehot.scatter()
	return 1 - dice;
}

//prediction [NCHW], target [NHW]
torch::Tensor CELoss(torch::Tensor prediction, torch::Tensor target) {
	return torch::nll_loss2d(torch::log_softmax(prediction, /*dim=*/1), target);
}