#ifndef NM_LAYER_LIST_HH
#define NM_LAYER_LIST_HH

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <glog/logging.h>

using std::string, std::vector, std::pair;
using rapidjson::Document, rapidjson::PrettyWriter, rapidjson::StringBuffer;

class NM_LAYER_LIST {
  public:
    NM_LAYER_LIST(): layer_num(0) {};
    ~NM_LAYER_LIST() {};

    int loads(const char *json_str) {
        try {
            Document document;
        } catch (const std::exception &e) {
            LOG(ERROR) << "json_str parsed error " << e.what();
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    int loadf(const char *filename);
    int dumps(string &json_str) {
        try {  // https://medium.com/codeflu/understanding-rapidjson-e7fbf62492ba
            StringBuffer s;
            Document d;
            PrettyWriter<StringBuffer> writer(s);
            writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
            writer.StartArray();
            for(auto& layer_info: layer_list) {
                writer.StartArray();
                for(pair<string, string>& kv: layer_info) {
                    writer.StartArray();
                    writer.String(kv.first.c_str());
                    writer.String(kv.second.c_str());
                    writer.EndArray();
                }
                writer.EndArray();
            }
            writer.EndArray();

            json_str = string(s.GetString());
        } catch (const std::exception &e) {
            LOG(ERROR) << "Error: " << e.what();
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int dumpf(const char *filename) {
        try {
            std::string json_str;
            int dumps_rcode = dumps(json_str);
            if(dumps_rcode == EXIT_FAILURE) {
                throw std::runtime_error("NM_LAYER_LIST.dumps() error");
            }
            std::ofstream ofile(filename, std::ios::out);
            ofile << json_str;
            ofile.close();
        }
        catch (const std::exception &e) {
            LOG(ERROR) << "Error: " << e.what();
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int amend_layer_info(const vector<pair<string, string> > &layer_info) {
        if(layer_list.size() <= 0) {
            LOG(WARNING) << "amend an uninitialized vector \n";
            return EXIT_FAILURE;
        }
        auto &cur_layer_info = layer_list.back();
        cur_layer_info.insert(cur_layer_info.end(), layer_info.begin(), layer_info.end());
        return EXIT_SUCCESS;
    }

    int add_layer_info(const vector<pair<string, string> > &layer_info) {
        layer_list.push_back(vector<pair<string, string> >({{"id", std::to_string(layer_num)}}));
      ++layer_num;
        amend_layer_info(layer_info);
      return EXIT_SUCCESS;
    }

  public:
    std::vector<vector<pair<string, string>>> layer_list;
    int layer_num;
};
#endif