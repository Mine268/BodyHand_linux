#include <algorithm>
#include "HaMeRONNX.h"

namespace BodyHand {

    cv::Mat cropAndResize(const cv::Mat img, float x1, float x2, float y1, float y2, double& scale_factor, int target_size) {
        float box_center_x = (x1 + x2) / 2;
        float box_center_y = (y1 + y2) / 2;
        float box_width = (x2 - x1);
        float box_height = (y2 - y1);

        float origin_square_size = int(std::max(box_width, box_height));
        origin_square_size *= 2; // ��Χ�����䵽����
        int new_x1 = int(box_center_x - origin_square_size / 2);
        int new_x2 = int(box_center_x + origin_square_size / 2);
        int new_y1 = int(box_center_y - origin_square_size / 2);
        int new_y2 = int(box_center_y + origin_square_size / 2);

        cv::Mat origin_square_img(origin_square_size, origin_square_size, img.type());
        int src_x1 = std::max(new_x1, 0);
        int src_y1 = std::max(new_y1, 0);
        int src_x2 = std::min(new_x2, img.cols - 1);
        int src_y2 = std::min(new_y2, img.rows - 1);
        int dst_x = src_x1 - new_x1;
        int dst_y = src_y1 - new_y1;
        cv::Rect srcRect(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1);
        cv::Rect dstRect(dst_x, dst_y, srcRect.width, srcRect.height);

        if (srcRect.width > 0 && srcRect.height > 0 &&
            dstRect.x >= 0 && dstRect.y >= 0 &&
            dstRect.x + dstRect.width <= origin_square_img.cols &&
            dstRect.y + dstRect.height <= origin_square_img.rows)
        {
            img(srcRect).copyTo(origin_square_img(dstRect));
        }
        else {
            throw std::runtime_error("Cannot crop & resize this image");
        }

        cv::Mat output_square;
        scale_factor = origin_square_size / float(target_size);
        cv::resize(origin_square_img, output_square, cv::Size(target_size, target_size));
        return output_square;
    }

	HaMeROnnx::HaMeROnnx() :
		handlr_session(nullptr),
		hamer_session(nullptr)
	{
		ort_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	}

	bool HaMeROnnx::loadModel(std::string _handlr_model_path, std::string _hamer_model_path) {
		handlr_model_path = _handlr_model_path;
		hamer_model_path = _hamer_model_path;
		try {
			std::string handlr_model_path_w(handlr_model_path.begin(), handlr_model_path.end());
			std::string hamer_model_path_w(hamer_model_path.begin(), hamer_model_path.end());

            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            ort_opt.AppendExecutionProvider_CUDA(cudaOption);

			handlr_session = new Ort::Session(
				ort_env,
				handlr_model_path_w.c_str(),
				ort_opt
			);
			hamer_session = new Ort::Session(
				ort_env,
				hamer_model_path_w.c_str(),
				ort_opt
			);
		}
		catch (const Ort::Exception& e) {
			delete this->handlr_session;
			delete this->hamer_session;
			return false;
		}
		catch (const std::exception& e) {
			delete this->handlr_session;
			delete this->hamer_session;
			return false;
		}
		return true;
	}

    std::vector<float> HaMeROnnx::postProcessHandPosition(std::vector<Ort::Value>& output_tensor, int lr_flag) {
        std::vector<float> hand_position(126, 0);
        Ort::Value& output_joint_positions_tensor = output_tensor[0];  // ��ȡ������� joint_positions
        auto tensor_hand_info = output_joint_positions_tensor.GetTensorTypeAndShapeInfo();
        auto tensor_hand_shape = tensor_hand_info.GetShape();  // ��ȡ������ά��
        auto data_ptr = output_joint_positions_tensor.GetTensorData<float>();  // ��ȡ��������ָ��
        if (data_ptr != nullptr) {
            size_t data_size = tensor_hand_info.GetElementCount();  // ��ȡԪ������
            for (int index = 0; index < data_size / 3; ++index) {
                if (lr_flag == 0) hand_position[63 * lr_flag + index * 3] = -1 * data_ptr[index * 3 + 1]; // ���ַ�ת x :
                else if (lr_flag == 1) hand_position[63 * lr_flag + index * 3] = data_ptr[index * 3 + 1]; // ���� x
                hand_position[63 * lr_flag + index * 3 + 1] = data_ptr[index * 3 + 2];   // y : z
                hand_position[63 * lr_flag + index * 3 + 2] = data_ptr[index * 3];     // z : x
            }
        }
        return hand_position;
    }

    std::tuple<std::vector<float>, std::vector<float>>
    HaMeROnnx::postProcessHandRotation(std::vector<Ort::Value>& output_tensor, int lr_flag) {
        std::vector<float> hand_rotation(288, 0);
        std::vector<float> hand_rotation_aa(96, 0);
        Ort::Value& output_joint_rotations_tensor = output_tensor[1];  // ��ȡ������� joint_rotations
        auto tensor_rotations_info = output_joint_rotations_tensor.GetTensorTypeAndShapeInfo();
        auto tensor_rotations_shape = tensor_rotations_info.GetShape();  // ��ȡ������ά��
        auto rotations_ptr = output_joint_rotations_tensor.GetTensorData<float>();  // ��ȡ��������ָ��
        if (rotations_ptr != nullptr) {
            size_t rotations_size = tensor_rotations_info.GetElementCount();  // ��ȡԪ������
            for (int index = 0; index < rotations_size; ++index) {
                hand_rotation[144 * lr_flag + index] = rotations_ptr[index];
            }
        }
        if (lr_flag == 0) { // ���ַ�ת
            for (int index = 0; index < 16; ++index) {
                // ��ȡ 3��3 ��ת����
                cv::Mat R = cv::Mat(3, 3, CV_32F, &hand_rotation[9 * index]).clone(); // ����ֱ���޸� handRotation
                // ��Ǳ�ʾ
                cv::Mat axis_angle;
                cv::Rodrigues(R, axis_angle);
                // ����ˮƽ��ת (x, -y, -z)
                cv::Mat flip_axis_angle = (cv::Mat_<float>(3, 1) <<
                    axis_angle.at<float>(0, 0),
                    -axis_angle.at<float>(1, 0),
                    -axis_angle.at<float>(2, 0));

                hand_rotation_aa[3 * index + 0] = flip_axis_angle.at<float>(0, 0);
                hand_rotation_aa[3 * index + 1] = flip_axis_angle.at<float>(1, 0);
                hand_rotation_aa[3 * index + 2] = flip_axis_angle.at<float>(2, 0);
                // ����ת��Ϊ��ת����
                cv::Mat flipped_R;
                cv::Rodrigues(flip_axis_angle, flipped_R);
                // ��� handRotation ����
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        hand_rotation[9 * index + i * 3 + j] = flipped_R.at<float>(i, j);
                    }
                }
            }
        }
        else {
            for (int index = 0; index < 16; ++index) {
                // ��ȡ 3��3 ��ת����
                cv::Mat R = cv::Mat(3, 3, CV_32F, &hand_rotation[144 + 9 * index]).clone(); // ����ֱ���޸� handRotation
                // ��Ǳ�ʾ
                cv::Mat axis_angle;
                cv::Rodrigues(R, axis_angle);
                hand_rotation_aa[48 + 3 * index + 0] = axis_angle.at<float>(0, 0);
                hand_rotation_aa[48 + 3 * index + 1] = axis_angle.at<float>(1, 0);
                hand_rotation_aa[48 + 3 * index + 2] = axis_angle.at<float>(2, 0);
            }
        }
        return { hand_rotation, hand_rotation_aa };
    }

    std::vector<float> HaMeROnnx::postProcessHandShape(std::vector<Ort::Value>& output_tensor, int lr_flag) {
        std::vector<float> hand_shape(20, 0);
        Ort::Value& output_hand_shape_tensor = output_tensor[2];  // ��ȡ������� hand_shape
        auto tensor_shape_info = output_hand_shape_tensor.GetTensorTypeAndShapeInfo();
        auto tensor_shape_shape = tensor_shape_info.GetShape();  // ��ȡ������ά��
        auto shape_ptr = output_hand_shape_tensor.GetTensorData<float>();  // ��ȡ��������ָ��
        if (shape_ptr != nullptr) {
            size_t shape_size = tensor_shape_info.GetElementCount(); // ��ȡԪ������
            for (int index = 0; index < shape_size; ++index) {
                hand_shape[10 * lr_flag + index] = shape_ptr[index];
            }
        }
        return hand_shape;
    }

    std::vector<float> HaMeROnnx::postProcessHand2D(std::vector<Ort::Value>& output_tensor, int lr_flag) {
        std::vector<float> hand_2d(84, 0);
        Ort::Value& output_joint2d_tensor = output_tensor[4];  // ��ȡ������� hand_shape
        auto tensor_joint2d_info = output_joint2d_tensor.GetTensorTypeAndShapeInfo();
        auto tensor_joint2d_shape = tensor_joint2d_info.GetShape();  // ��ȡ������ά��
        auto joint2d_ptr = output_joint2d_tensor.GetTensorData<float>();  // ��ȡ��������ָ��
        if (joint2d_ptr != nullptr) {
            size_t joints2d_size = tensor_joint2d_info.GetElementCount();  // ��ȡԪ������
            for (int i = 0; i < 21; ++i) {
                hand_2d[42 * lr_flag + i * 2] = (2 * lr_flag - 1) * joint2d_ptr[i * 2];
                hand_2d[42 * lr_flag + i * 2 + 1] = joint2d_ptr[i * 2 + 1];
            }
        }
        return hand_2d;
    }

    std::tuple<
        std::vector<float>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>
    > HaMeROnnx::postProcessOutputTensor(std::vector<Ort::Value>& output_tensors, int lr_flag) {
        std::vector<float> hand_position = postProcessHandPosition(output_tensors, lr_flag);
        auto [hand_rotation, hand_rotation_aa] = postProcessHandRotation(output_tensors, lr_flag);
        std::vector<float> hand_shape = postProcessHandShape(output_tensors, lr_flag);
        std::vector<float> hand_2d = postProcessHand2D(output_tensors, lr_flag);
        return {
            hand_position,
            hand_rotation,
            hand_rotation_aa,
            hand_shape,
            hand_2d
        };
    }

    std::tuple<std::vector<float>, std::vector<cv::Point3f>, std::vector<cv::Point2f>>
    HaMeROnnx::restorePose(
        std::vector<float>& hand_position,
        std::vector<float>& hand_2d,
        int lr_flag,
        float scale_factor,
        cv::Point2f center
    ) {
        std::vector<float> hand_2d_restore(84, 0);
        std::vector<cv::Point3f> hand_position_cv(21);
        std::vector<cv::Point2f> hand_2d_cv(21);
        for (int i = 0; i < 21; i++) {
            hand_2d_restore[42 * lr_flag + i * 2] = (256 * hand_2d[42 * lr_flag + i * 2]) * scale_factor + center.x;
            hand_2d_restore[42 * lr_flag + i * 2 + 1] = (256 * hand_2d[42 * lr_flag + i * 2 + 1]) * scale_factor + center.y;
            hand_position_cv[i] = cv::Point3f(hand_position[63 * lr_flag + 3 * i], hand_position[63 * lr_flag + 3 * i + 1], hand_position[63 * lr_flag + 3 * i + 2]);
            hand_position_cv[i] -= cv::Point3f(hand_position[63 * lr_flag + 0], hand_position[63 * lr_flag + 1], hand_position[63 * lr_flag + 2]);
            hand_2d_cv[i] = cv::Point2f(hand_2d_restore[42 * lr_flag + 2 * i], hand_2d_restore[42 * lr_flag + 2 * i + 1]);
        }
        return {
            hand_2d_restore,
            hand_position_cv,
            hand_2d_cv
        };
    }

    std::vector<cv::Point3f> HaMeROnnx::recoverPoseCam(
        std::vector<cv::Point3f> local_pos,
        std::vector<cv::Point2f> img_coord,
        const cv::Mat& intr,
        const std::vector<float> undist
    ) {
        cv::Mat rvec, tvec;
        bool flag = cv::solvePnPRansac(
            local_pos,
            img_coord,
            intr,
            undist,
            rvec,
            tvec,
            false
        );

        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Mat Rf, tvecf;
        R.convertTo(Rf, CV_32F);
        tvec.convertTo(tvecf, CV_32F);
        std::vector<cv::Point3f> cam_pos;
        for (const auto& point : local_pos) {
            cv::Mat pointMat = (cv::Mat_<float>(3, 1) << point.x, point.y, point.z);
            cv::Mat transformedMat = Rf * pointMat + tvecf;
            cam_pos.emplace_back(transformedMat.at<float>(0), transformedMat.at<float>(1), transformedMat.at<float>(2));
        }
        return cam_pos;
    }

    std::vector<const char*> HaMeROnnx::getInputNodeNames(Ort::Session* session) {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session->GetInputCount();
        std::vector<const char*> input_names;
        input_names.reserve(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name_alloc = session->GetInputNameAllocated(i, allocator);
            std::string name_str = input_name_alloc.get();
            char* name_copy = new char[name_str.size() + 1];
            strcpy(name_copy, name_str.c_str());
            input_names.push_back(name_copy);
        }
        return input_names;
    }

    std::vector<const char*> HaMeROnnx::getOutputNodeNames(Ort::Session* session) {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_output_nodes = session->GetOutputCount();
        std::vector<const char*> output_names;
        output_names.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name_alloc = session->GetOutputNameAllocated(i, allocator);
            std::string name_str = output_name_alloc.get();
            char* name_copy = new char[name_str.size() + 1];
            strcpy(name_copy, name_str.c_str());
            output_names.push_back(name_copy);
        }
        return output_names;
    }

    std::vector<Ort::Value> HaMeROnnx::detectHandBox(cv::Mat& img) {
        cv::Mat resized_image;
        cv::resize(img, resized_image, cv::Size(640, 480));
        //cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);  //BGR -> RGB
        std::vector<int64_t> input_shape = { 1, 3, 480, 640 };
        std::vector<float> input_data(1 * 3 * 480 * 640);
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 480; h++)
                for (int w = 0; w < 640; w++)
                    input_data[c * 480 * 640 + h * 640 + w] = resized_image.at<cv::Vec3b>(h, w)[c];
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );
        //std::vector<const char*> input_node_names = { "input" };
        std::vector<const char*> input_node_names = getInputNodeNames(handlr_session);
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        //std::vector<const char*> output_node_names = { "batchno_classid_score_x1y1x2y2" };
        std::vector<const char*> output_node_names = getOutputNodeNames(handlr_session);

        std::vector<Ort::Value> output_tensors = handlr_session->Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(),
            input_tensors.data(),
            1,
            output_node_names.data(),
            1
        );

        return output_tensors;
    }

    std::vector<Ort::Value> HaMeROnnx::detectHandFromBox(cv::Mat& hand_img) {
        std::vector<int64_t> input_hand_shape = { 1, 3, 256, 256 };
        std::vector<float> input_hand_data(1 * 3 * 256 * 256, 1.0f); // �������
        // ��������䵽 input_hand_data ��
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < hand_img.rows; ++h) {
                for (int w = 0; w < hand_img.cols; ++w) {
                    input_hand_data[c * 256 * 256 + h * 256 + w] = hand_img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                }
            }
        }
        const OrtMemoryInfo* _memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value input_hand_tensor = Ort::Value::CreateTensor<float>(
            _memory_info,
            input_hand_data.data(),       // ��������
            input_hand_data.size(),       // ���ݵĴ�С
            input_hand_shape.data(),      // �������״
            input_hand_shape.size()       // ��״��ά������
        );
        //std::vector<const char*> input_hand_node_names = { "input" }; // ����ڵ�����
        std::vector<const char*> input_hand_node_names = getInputNodeNames(hamer_session);
        std::vector<Ort::Value> input_hand_tensors;

        input_hand_tensors.push_back(std::move(input_hand_tensor));
        //std::vector<const char*> output_hand_node_names = { "joint_positions", "joint_rotations", "hand_shape", "camera_trans", "joint_img" }; // ����ڵ�����
        std::vector<const char*> output_hand_node_names = getOutputNodeNames(hamer_session);
        auto output_hand_tensors = hamer_session->Run(
            Ort::RunOptions{ nullptr },
            input_hand_node_names.data(),
            input_hand_tensors.data(),
            input_hand_tensors.size(),
            output_hand_node_names.data(),
            5
        );

        return output_hand_tensors;
    }

    std::tuple<bool, bool> HaMeROnnx::detectPose(
        cv::Mat& img,
        const cv::Mat intr,
        const std::vector<float> undist,
        std::vector<cv::Point3f>& hand_position_cam,
        std::vector<cv::Point2f>& hand_position_2d,
        std::optional<std::reference_wrapper<std::vector<cv::Rect2f>>> hand_bbox
    ) {
        hand_position_cam.clear();
        hand_position_cam.resize(42);
        hand_position_2d.clear();
        hand_position_2d.resize(42);
        if (hand_bbox) {
            std::vector<cv::Rect2f>& vec = hand_bbox->get();
            vec.resize(2);
        }
        bool valid_right{ false }, valid_left{ false };

        double scale_factor = 1.0f;
        auto output_tensors = detectHandBox(img);
        float scale_width = static_cast<float>(img.cols) / 640;
        float scale_height = static_cast<float>(img.rows) / 480;
        //cv::Point scale(scale_width, scale_height);
        cv::Point2f scale(2.25, 2.25);
        //std::cout << scale << std::endl;
        auto& output_tensor = output_tensors[0];
        float* output_data = output_tensor.GetTensorMutableData<float>();
        std::vector<int64_t>  output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        size_t boxes_num = output_shape[0];
        for (size_t box_ind = 0; box_ind < boxes_num; ++box_ind) {
            float classid = output_data[box_ind * 7 + 1]; // ���ID
            float score = output_data[box_ind * 7 + 2]; // ���Ŷ�
            float x1 = output_data[box_ind * 7 + 3] * scale_width; // x1
            float y1 = output_data[box_ind * 7 + 4] * scale_height; // y1
            float x2 = output_data[box_ind * 7 + 5] * scale_width; // x2
            float y2 = output_data[box_ind * 7 + 6] * scale_height; // y2
            cv::Point2f center((x2 + x1) / 2, (y2 + y1) / 2);
            if (score < 0.5) continue;
            // debug
            //cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(40 * classid, 255 - 40 * classid, 0), 2);
            std::vector<Ort::Value> hand_output_tensors;
            switch (int(classid))
            {
            case 4: //����
            {
                if (hand_bbox) {
                    std::vector<cv::Rect2f>& vec = hand_bbox->get();
                    vec[0] = cv::Rect2f{x1, y1, x2 - x1, y2 - y1};
                }
                cv::Mat left_image = cropAndResize(img, x1, x2, y1, y2, scale_factor, 256);
                //std::cout << scale_factor << std::endl;
                cv::flip(left_image, left_image, 1); //�� y�� ��ת  ����ת
                hand_output_tensors = detectHandFromBox(left_image);
                auto [hand_position, hand_rotation, hand_rotation_aa, hand_shape, hand_2d] =
                    postProcessOutputTensor(hand_output_tensors, 0);
                auto [hand_2d_restore, hand_position_cv, hand_2d_cv] =
                    restorePose(hand_position, hand_2d, 0, scale_factor, center);

                std::vector<cv::Point3f> left_hand_cam =
                    recoverPoseCam(hand_position_cv, hand_2d_cv, intr, undist);

                std::copy(left_hand_cam.begin(), left_hand_cam.end(), hand_position_cam.begin());
                std::copy(hand_2d_cv.begin(), hand_2d_cv.end(), hand_position_2d.begin());
                valid_left = true;
            }
            break;
            case 5://����
            {
                if (hand_bbox) {
                    std::vector<cv::Rect2f>& vec = hand_bbox->get();
                    vec[1] = cv::Rect2f{ x1, y1, x2 - x1, y2 - y1 };
                }
                cv::Mat right_image = cropAndResize(img, x1, x2, y1, y2, scale_factor, 256);
                hand_output_tensors = detectHandFromBox(right_image);
                auto [hand_position, hand_rotation, hand_rotation_aa, hand_shape, hand_2d ] =
                    postProcessOutputTensor(hand_output_tensors, 1);
                auto [ hand_2d_restore, hand_position_cv, hand_2d_cv] =
                    restorePose(hand_position, hand_2d, 1, scale_factor, center);

                std::vector<cv::Point3f> right_hand_cam =
                    recoverPoseCam(hand_position_cv, hand_2d_cv, intr, undist);

                std::copy(right_hand_cam.begin(), right_hand_cam.end(), hand_position_cam.begin() + 21);
                std::copy(hand_2d_cv.begin(), hand_2d_cv.end(), hand_position_2d.begin() + 21);
                valid_right = true;
            }
            break;
            default:
                break;
            }
        }
        // ��λ���ױ�ɺ���
        std::transform(
            hand_position_cam.begin(), hand_position_cam.end(),
            hand_position_cam.begin(),
            [](auto& p) { return 1e3 * p; }
        );
        return { valid_left, valid_right };
    }
}