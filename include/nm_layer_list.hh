#ifndef NM_LAYER_LIST_HH
#define NM_LAYER_LIST_HH

#include <vector>
#include <string>
#include <iostream>
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
        try {
            StringBuffer s;
            Document d;
            PrettyWriter<StringBuffer> writer(s);
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

//            std::cout << "result: \n" << s.GetString() << "\n";
        } catch (const std::exception &e) {
            LOG(ERROR) << "Error: " << e.what();
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int dumpf(const char *filename) {
        std::string json_str;
        try {
            int ret_code = dumps(json_str);
        }
        catch (const std::exception &e) {
            LOG(ERROR) << "Error: " << e.what();
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int add_layer_info(const vector<pair<string, string> > &layer_info) {
        layer_list.push_back(vector<pair<string, string> >({{"id", std::to_string(layer_num)}}));
        auto &cur_layer_info = layer_list.back();
        cur_layer_info.insert(cur_layer_info.end(), layer_info.begin(), layer_info.end());
      ++layer_num;
      return 0;
    }

    int admend_layer_info();

  public:
    std::vector<vector<pair<string, string>>> layer_list;
    int layer_num;
};
#endif