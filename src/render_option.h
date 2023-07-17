#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

struct RenderOption {
    int width = 800;
    float aspect_ratio = 16.0f / 9.0f;
    int max_depth = 50;
    int num_samples = 100;
    bool use_gpu = false;
    std::string output_file = "output.png";

    static RenderOption parseCommandLine(int argc, char** argv) {
        RenderOption opt;
        for (int i = 1; i < argc; ++i) {
            try {
                if (std::string(argv[i]) == "--width" || std::string(argv[i]) == "-w") {
                    opt.width = std::stoi(argv[++i]);
                } else if (std::string(argv[i]) == "--aspect_ratio" || std::string(argv[i]) == "-a") {
                    opt.aspect_ratio = std::stof(argv[++i]);
                } else if (std::string(argv[i]) == "--max_depth" || std::string(argv[i]) == "-d") {
                    opt.max_depth = std::stoi(argv[++i]);
                } else if (std::string(argv[i]) == "--num_samples" || std::string(argv[i]) == "-n") {
                    opt.num_samples = std::stoi(argv[++i]);
                } else if (std::string(argv[i]) == "--use_gpu" || std::string(argv[i]) == "-g") {
                    opt.use_gpu = true;
                } else if (std::string(argv[i]) == "--output_file" || std::string(argv[i]) == "-o") {
                    opt.output_file = argv[++i];
                    // make sure it ends with .png
                    if (opt.output_file.size() < 4 || opt.output_file.substr(opt.output_file.size() - 4) != ".png") {
                        opt.output_file += ".png";
                    }
                } else {
                    throw std::invalid_argument("Unknown command line argument");
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing command line argument: " << argv[i] << std::endl;
                std::cout << "Valid arguments: " << std::endl;
                std::cout << "  --width, -w <int>" << std::endl;
                std::cout << "  --aspect_ratio, -a <float>" << std::endl;
                std::cout << "  --max_depth, -d <int>" << std::endl;
                std::cout << "  --num_samples, -n <int>" << std::endl;
                std::cout << "  --use_gpu, -g" << std::endl;
                std::cout << "  --output_file, -o <string>" << std::endl;
                throw new std::invalid_argument("Invalid command line argument");
            }
        }

        return opt;
    }
};

inline std::ostream& operator<<(std::ostream& os, const RenderOption& opt) {
    os << "RenderOption: " << std::endl;
    os << "  width: " << opt.width << std::endl;
    os << "  aspect_ratio: " << opt.aspect_ratio << std::endl;
    os << "  max_depth: " << opt.max_depth << std::endl;
    os << "  num_samples: " << opt.num_samples << std::endl;
    os << "  use_gpu: " << opt.use_gpu << std::endl;
    os << "  output_file: " << opt.output_file << std::endl;
    return os;
}