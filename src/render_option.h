#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

struct RenderOption {
    int width = 1200;
    int height = 800;
    int max_depth = 50;
    int num_samples = 10;
    bool use_gpu = false;
    std::string output_file = "output.ppm";

    static RenderOption parseCommandLine(int argc, char **argv) {
        RenderOption opt;
        for (int i = 1; i < argc; ++i) {
            try {
                if (std::string(argv[i]) == "--width" ||
                    std::string(argv[i]) == "-w") {
                    opt.width = std::stoi(argv[++i]);
                } else if (std::string(argv[i]) == "--height" ||
                           std::string(argv[i]) == "-h") {
                    opt.height = std::stof(argv[++i]);
                } else if (std::string(argv[i]) == "--max_depth" ||
                           std::string(argv[i]) == "-d") {
                    opt.max_depth = std::stoi(argv[++i]);
                } else if (std::string(argv[i]) == "--num_samples" ||
                           std::string(argv[i]) == "-n") {
                    opt.num_samples = std::stoi(argv[++i]);
                } else if (std::string(argv[i]) == "--use_gpu" ||
                           std::string(argv[i]) == "-g") {
                    opt.use_gpu = true;
                } else if (std::string(argv[i]) == "--output_file" ||
                           std::string(argv[i]) == "-o") {
                    opt.output_file = argv[++i];
                    // make sure it ends with .ppm
                    if (opt.output_file.size() < 4 ||
                        opt.output_file.substr(opt.output_file.size() - 4) !=
                            ".ppm") {
                        opt.output_file += ".ppm";
                    }
                } else {
                    throw std::invalid_argument(
                        "Unknown command line argument");
                }
            } catch (const std::exception &e) {
                std::cerr << "Error parsing command line argument: " << argv[i]
                          << std::endl;
                std::cout << "Valid arguments: " << std::endl;
                std::cout << "  --width, -w <int>" << std::endl;
                std::cout << "  --height, -h <int>" << std::endl;
                std::cout << "  --max_depth, -d <int>" << std::endl;
                std::cout << "  --num_samples, -n <int>" << std::endl;
                std::cout << "  --use_gpu, -g" << std::endl;
                std::cout << "  --output_file, -o <string>" << std::endl;
                throw new std::invalid_argument(
                    "Invalid command line argument");
            }
        }

        return opt;
    }
};

inline std::ostream &operator<<(std::ostream &os, const RenderOption &opt) {
    os << "RenderOption: " << std::endl;
    os << "  width: " << opt.width << std::endl;
    os << "  height: " << opt.height << std::endl;
    os << "  max_depth: " << opt.max_depth << std::endl;
    os << "  num_samples: " << opt.num_samples << std::endl;
    os << "  use_gpu: " << opt.use_gpu << std::endl;
    os << "  output_file: " << opt.output_file << std::endl;
    return os;
}