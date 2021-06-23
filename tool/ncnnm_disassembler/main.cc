#include <iostream>
#include <fstream>

#include <vector>
#include <string>

#include <glog/logging.h>

#include <filesystem>

#include "ncnn/net.h"
#include "ncnn/layer_type.h"
#include "ncnn/datareader.h"  // DataReaderFromStdio

std::string raw_model_profile_path("raw_model_profile.txt");  // name of raw_profile_path

//将信息输出到单独的文件和 LOG(ERROR)
void SignalHandle(const char *data, int size) {
  std::ofstream fs("glog_dump.log", std::ios::app);
  std::string str = std::string(data, size);
  fs << str;
  fs.close();
  LOG(ERROR) << str;
}

class GLogHelper {
 public:
  GLogHelper(const char *program) {
    google::InitGoogleLogging(program);
    FLAGS_colorlogtostderr = true;
    google::InstallFailureSignalHandler();
    //默认捕捉 SIGSEGV 信号信息输出会输出到 stderr，可以通过下面的方法自定义输出方式：
    google::InstallFailureWriter(&SignalHandle);

    google::SetLogDestination(google::GLOG_INFO, "./result_");
  }

  ~GLogHelper() {
    google::ShutdownGoogleLogging();
  }
};

namespace ncnn_M {

class ModelBinFromDataReaderPrivate {
 public:
  ModelBinFromDataReaderPrivate(const ncnn::DataReader &_dr)
      :
      dr(_dr) {
  }
  const ncnn::DataReader &dr;
};

// rewrite this class (specially .load()) to export ncnn_bin contents
class ModelBinFromDataReader : public ncnn::ModelBin {
 public:
  explicit ModelBinFromDataReader(const ncnn::DataReader &dr);
  virtual ~ModelBinFromDataReader();

  virtual ncnn::Mat load(int w, int type) const;  //

 private:
  ModelBinFromDataReader(const ModelBinFromDataReader&);
  ModelBinFromDataReader& operator=(const ModelBinFromDataReader&);

 private:
  ncnn_M::ModelBinFromDataReaderPrivate *const d;
};

ModelBinFromDataReader::ModelBinFromDataReader(const ncnn::DataReader &_dr)
    :
    ModelBin(),
    d(new ncnn_M::ModelBinFromDataReaderPrivate(_dr)) {
}

ModelBinFromDataReader::~ModelBinFromDataReader() {
  delete d;
}

ModelBinFromDataReader::ModelBinFromDataReader(const ModelBinFromDataReader&)
    :
    d(0) {
}

ModelBinFromDataReader& ModelBinFromDataReader::operator=(
    const ModelBinFromDataReader&) {
  return *this;
}

ncnn::Mat ModelBinFromDataReader::load(int w, int type) const {  // read layer's weights from xxmodel.bin
  std::ofstream ofile(raw_model_profile_path, std::ios::out | std::ios::app);
  ofile << "    ModelBinFromDataReader->load(" << w << ", " << type << ")";

  if (type == 0) {  // todo: 一句话解释一下这个
    size_t nread;

    union {
      struct {
        unsigned char f0;
        unsigned char f1;
        unsigned char f2;
        unsigned char f3;
      };
      unsigned int tag;
    } flag_struct;

    nread = d->dr.read(&flag_struct, sizeof(flag_struct));   // 先读flag_struct
    ofile << "seg(" << sizeof(flag_struct) << ") ";
    if (nread != sizeof(flag_struct)) {
      NCNN_LOGE("ModelBin read flag_struct failed %zd", nread);
      ofile << "\n";
      ofile.close();
      return ncnn::Mat();
    }

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2
        + flag_struct.f3;  // 我去，还得相加。。。生成flag

    if (flag_struct.tag == 0x01306B47)  // 应该是float16相关
        {
      // half-precision data
      size_t align_data_size = ncnn::alignSize(w * sizeof(unsigned short), 4);
      std::vector<unsigned short> float16_weights;
      float16_weights.resize(align_data_size);
      nread = d->dr.read(float16_weights.data(), align_data_size);   // 实际读取数量
      ofile << "tag == 0x01306B47 seg(" << align_data_size << ") ";

      if (nread != align_data_size) {
        NCNN_LOGE("ModelBin read float16_weights failed %zd", nread);
        ofile << "\n";
        ofile.close();
        return ncnn::Mat();
      }

      ofile << "\n";
      ofile.close();
      return ncnn::Mat::from_float16(float16_weights.data(), w);
    } else if (flag_struct.tag == 0x000D4B38)  // 应该是int8相关
        {
      // int8 data
      size_t align_data_size = ncnn::alignSize(w, 4);
      std::vector<signed char> int8_weights;
      int8_weights.resize(align_data_size);
      nread = d->dr.read(int8_weights.data(), align_data_size);    // 实际读取数量
      ofile << "tag == 0x000D4B38 seg(" << align_data_size << ") ";
      if (nread != align_data_size) {
        NCNN_LOGE("ModelBin read int8_weights failed %zd", nread);
        ofile << "\n";
        ofile.close();
        return ncnn::Mat();
      }

      ncnn::Mat m(w, (size_t) 1u);
      if (m.empty()) {      // 难道是无法分配出来?
        ofile << "\n";
        ofile.close();
        return m;
      }

      memcpy(m.data, int8_weights.data(), w);

      return m;
    } else if (flag_struct.tag == 0x0002C056)  // 应该是正常？读入的也是没有align的数值
        {
      ncnn::Mat m(w);
      if (m.empty()) {
        ofile << "\n";
        ofile.close();
        return m;
      }

      // raw data with extra scaling
      nread = d->dr.read(m, w * sizeof(float));
      ofile << "tag == 0x0002C056 seg(" << w * sizeof(float) << ") ";
      if (nread != w * sizeof(float)) {
        NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
        ofile << "\n";
        ofile.close();
        return ncnn::Mat();
      }

      ofile << "\n";
      ofile.close();
      return m;
    }

    ncnn::Mat m(w);
    if (m.empty()) {
      ofile << "\n";
      ofile.close();
      return m;
    }

    if (flag != 0)    // 要做量化
        {
      // quantized data
      float quantization_value[256];    // 量化表
      nread = d->dr.read(quantization_value, 256 * sizeof(float));
      ofile << "flag!=0 seg(" << 256 * sizeof(float) << ") ";
      if (nread != 256 * sizeof(float)) {
        NCNN_LOGE("ModelBin read quantization_value failed %zd", nread);
        ofile << "\n";
        ofile.close();
        return ncnn::Mat();
      }

      size_t align_weight_data_size = ncnn::alignSize(w * sizeof(unsigned char),
                                                      4);
      std::vector<unsigned char> index_array;
      index_array.resize(align_weight_data_size);
      nread = d->dr.read(index_array.data(), align_weight_data_size);
      ofile << "seg(" << align_weight_data_size << ") ";
      if (nread != align_weight_data_size) {
        NCNN_LOGE("ModelBin read index_array failed %zd", nread);
        ofile << "\n";
        ofile.close();
        return ncnn::Mat();
      }

      float *ptr = m;
      for (int i = 0; i < w; i++) {
        ptr[i] = quantization_value[index_array[i]];
      }
    } else if (flag_struct.f0 == 0) {
      // raw data
      nread = d->dr.read(m, w * sizeof(float));    // 也是读取没有align的输入数据
      ofile << "flag_struct.f0 == 0 seg(" << w * sizeof(float) << ") ";
      if (nread != w * sizeof(float)) {
        NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
        ofile << "\n";
        ofile.close();
        return ncnn::Mat();
      }
    }

    ofile << "\n";
    ofile.close();
    return m;
  } else if (type == 1) {
    ncnn::Mat m(w);   // 应该是w是0
    if (m.empty()) {
      ofile << "w = " << w << " early STOP\n";
      ofile.close();
      return m;
    }

    // raw data
    size_t nread = d->dr.read(m, w * sizeof(float));    // 读入的是直接float
    ofile << "type == 1 seg(" << w * sizeof(float) << ") ";
    if (nread != w * sizeof(float)) {
      NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
      ofile << "\n";
      ofile.close();
      return ncnn::Mat();
    }

    ofile << "\n";
    ofile.close();
    return m;
  } else  // 其他的type
  {
    NCNN_LOGE("ModelBin load type %d not implemented", type);
    ofile << "\n";
    ofile.close();
    return ncnn::Mat();
  }

  ofile << "\n";
  ofile.close();
  return ncnn::Mat();
}

class ParamDictPrivate {
 public:
  struct {
    // 0 = null
    // 1 = int/float
    // 2 = int
    // 3 = float
    // 4 = array of int/float
    // 5 = array of int
    // 6 = array of float
    int type;
    union {
      int i;
      float f;
    };
    ncnn::Mat v;
  } params[NCNN_MAX_PARAM_COUNT];
};

// 直接搬运ncnn的这个类，因为不晓得让friend对ncnn::Net继承类生效的问题
class ParamDict {
 public:
  // empty
  ParamDict();

  virtual ~ParamDict();

  // copy
  ParamDict(const ParamDict&);

  // assign
  ParamDict& operator=(const ParamDict&);

  // get type
  int type(int id) const;

  // get int
  int get(int id, int def) const;
  // get float
  float get(int id, float def) const;
  // get array
  ncnn::Mat get(int id, const ncnn::Mat &def) const;

  // set int
  void set(int id, int i);
  // set float
  void set(int id, float f);
  // set array
  void set(int id, const ncnn::Mat &v);

// protected:
  friend class Net;

  void clear();

  int load_param(const ncnn::DataReader &dr);
  int load_param_bin(const ncnn::DataReader &dr);

 private:
  ParamDictPrivate *const d;
};

ParamDict::ParamDict()
    :
    d(new ParamDictPrivate) {
  clear();
}

ParamDict::~ParamDict() {
  delete d;
}

ParamDict::ParamDict(const ParamDict &rhs)
    :
    d(new ParamDictPrivate) {
  for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++) {
    int type = rhs.d->params[i].type;
    d->params[i].type = type;
    if (type == 1 || type == 2 || type == 3) {
      d->params[i].i = rhs.d->params[i].i;
    } else  // if (type == 4 || type == 5 || type == 6)
    {
      d->params[i].v = rhs.d->params[i].v;
    }
  }
}

ParamDict& ParamDict::operator=(const ParamDict &rhs) {
  if (this == &rhs)
    return *this;

  for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++) {
    int type = rhs.d->params[i].type;
    d->params[i].type = type;
    if (type == 1 || type == 2 || type == 3) {
      d->params[i].i = rhs.d->params[i].i;
    } else  // if (type == 4 || type == 5 || type == 6)
    {
      d->params[i].v = rhs.d->params[i].v;
    }
  }

  return *this;
}

int ParamDict::type(int id) const {
  return d->params[id].type;
}

// TODO strict type check
int ParamDict::get(int id, int def) const {
  return d->params[id].type ? d->params[id].i : def;
}

float ParamDict::get(int id, float def) const {
  return d->params[id].type ? d->params[id].f : def;
}

ncnn::Mat ParamDict::get(int id, const ncnn::Mat &def) const {
  return d->params[id].type ? d->params[id].v : def;
}

void ParamDict::set(int id, int i) {
  d->params[id].type = 2;
  d->params[id].i = i;
}

void ParamDict::set(int id, float f) {
  d->params[id].type = 3;
  d->params[id].f = f;
}

void ParamDict::set(int id, const ncnn::Mat &v) {
  d->params[id].type = 4;
  d->params[id].v = v;
}

void ParamDict::clear() {
  for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++) {
    d->params[i].type = 0;
    d->params[i].v = ncnn::Mat();
  }
}

static bool vstr_is_float(const char vstr[16]) {
  // look ahead for determine isfloat
  for (int j = 0; j < 16; j++) {
    if (vstr[j] == '\0')
      break;

    if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
      return true;
  }

  return false;
}

static float vstr_to_float(const char vstr[16]) {
  double v = 0.0;

  const char *p = vstr;

  // sign
  bool sign = *p != '-';
  if (*p == '+' || *p == '-') {
    p++;
  }

  // digits before decimal point or exponent
  unsigned int v1 = 0;
  while (isdigit(*p)) {
    v1 = v1 * 10 + (*p - '0');
    p++;
  }

  v = (double) v1;

  // digits after decimal point
  if (*p == '.') {
    p++;

    unsigned int pow10 = 1;
    unsigned int v2 = 0;

    while (isdigit(*p)) {
      v2 = v2 * 10 + (*p - '0');
      pow10 *= 10;
      p++;
    }

    v += v2 / (double) pow10;
  }

  // exponent
  if (*p == 'e' || *p == 'E') {
    p++;

    // sign of exponent
    bool fact = *p != '-';
    if (*p == '+' || *p == '-') {
      p++;
    }

    // digits of exponent
    unsigned int expon = 0;
    while (isdigit(*p)) {
      expon = expon * 10 + (*p - '0');
      p++;
    }

    double scale = 1.0;
    while (expon >= 8) {
      scale *= 1e8;
      expon -= 8;
    }
    while (expon > 0) {
      scale *= 10.0;
      expon -= 1;
    }

    v = fact ? v * scale : v / scale;
  }

  //     fprintf(stderr, "v = %f\n", v);
  return sign ? (float) v : (float) -v;
}

int ParamDict::load_param(const ncnn::DataReader &dr) {
  clear();

  //     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

  // parse each key=value pair
  int id = 0;
  while (dr.scan("%d=", &id) == 1) {
    bool is_array = id <= -23300;
    if (is_array) {
      id = -id - 23300;
    }

    if (id >= NCNN_MAX_PARAM_COUNT) {
      NCNN_LOGE(
          "id < NCNN_MAX_PARAM_COUNT failed (id=%d, NCNN_MAX_PARAM_COUNT=%d)",
          id, NCNN_MAX_PARAM_COUNT);
      return -1;
    }

    if (is_array) {
      int len = 0;
      int nscan = dr.scan("%d", &len);
      if (nscan != 1) {
        NCNN_LOGE("ParamDict read array length failed");
        return -1;
      }

      d->params[id].v.create(len);

      for (int j = 0; j < len; j++) {
        char vstr[16];
        nscan = dr.scan(",%15[^,\n ]", vstr);
        if (nscan != 1) {
          NCNN_LOGE("ParamDict read array element failed");
          return -1;
        }

        bool is_float = vstr_is_float(vstr);

        if (is_float) {
          float *ptr = d->params[id].v;
          ptr[j] = vstr_to_float(vstr);
        } else {
          int *ptr = d->params[id].v;
          nscan = sscanf(vstr, "%d", &ptr[j]);
          if (nscan != 1) {
            NCNN_LOGE("ParamDict parse array element failed");
            return -1;
          }
        }

        d->params[id].type = is_float ? 6 : 5;
      }
    } else {
      char vstr[16];
      int nscan = dr.scan("%15s", vstr);
      if (nscan != 1) {
        NCNN_LOGE("ParamDict read value failed");
        return -1;
      }

      bool is_float = vstr_is_float(vstr);

      if (is_float) {
        d->params[id].f = vstr_to_float(vstr);
      } else {
        nscan = sscanf(vstr, "%d", &d->params[id].i);
        if (nscan != 1) {
          NCNN_LOGE("ParamDict parse value failed");
          return -1;
        }
      }

      d->params[id].type = is_float ? 3 : 2;
    }
  }

  return 0;
}

class NetPrivate {
 public:
  NetPrivate(ncnn::Option &_opt);

  ncnn::Option &opt;

  std::vector<ncnn::Blob> blobs;
  std::vector<ncnn::Layer*> layers;
  std::vector<ncnn::custom_layer_registry_entry> custom_layer_registry;
};

NetPrivate::NetPrivate(ncnn::Option &_opt)
    :
    opt(_opt) {
}

class Net : public ncnn::Net {
 public:
  Net();
  virtual ~Net();

  void clear();

  int load_param(const char *protopath);
  int load_param(FILE *fp);
  int load_param(const ncnn::DataReader &dr);

  int load_model(const char *modelpath);
  int load_model(FILE *fp);
  int load_model(const ncnn::DataReader &dr);

  int find_blob_index_by_name_MICRO(const char *name) const;

  ncnn::Option opt;
  ncnn_M::NetPrivate *d_;

 protected:
  virtual int custom_layer_to_index(const char *type);
  virtual ncnn::Layer* create_custom_layer(const char *type);
  virtual ncnn::Layer* create_custom_layer(int index);
};

Net::Net()
    :
    d_(new NetPrivate(opt)) {
}
Net::~Net() {
  clear();
  delete d_;

  // 避免进入父类的析构函数，再一次进入释放，造成问题，直接把程序hack退出
  // 实际程序结束的终点
  exit(EXIT_SUCCESS);
}

void Net::clear() {
  d_->blobs.clear();
  for (size_t i = 0; i < d_->layers.size(); i++) {
    ncnn::Layer *layer = d_->layers[i];

    ncnn::Option opt1 = opt;
    if (!layer->support_image_storage) {
      opt1.use_image_storage = false;
    }

    int dret = layer->destroy_pipeline(opt1);
    if (dret != 0) {
      NCNN_LOGE("layer destroy_pipeline failed");
      // ignore anyway
    }

    if (layer->typeindex & ncnn::LayerType::CustomBit) {
      int custom_index = layer->typeindex & ~ncnn::LayerType::CustomBit;
      if (d_->custom_layer_registry[custom_index].destroyer) {
        d_->custom_layer_registry[custom_index].destroyer(
            layer, d_->custom_layer_registry[custom_index].userdata);
      } else {
        delete layer;
      }
    } else {
      delete layer;
    }
  }
  d_->layers.clear();
}

int Net::find_blob_index_by_name_MICRO(const char *name) const {
  for (size_t i = 0; i < d_->blobs.size(); i++) {
    const ncnn::Blob &blob = d_->blobs[i];
    if (blob.name == name) {
      return static_cast<int>(i);
    }
  }

  NCNN_LOGE("find_blob_index_by_name %s failed", name);
  return -1;
}

int Net::load_param(const char *protopath) {
  FILE *fp = fopen(protopath, "rb");
  if (!fp) {
    NCNN_LOGE("fopen %s failed", protopath);
    return -1;
  }

  int ret = load_param(fp);
  fclose(fp);
  return ret;
}

int Net::load_param(FILE *fp) {
  ncnn::DataReaderFromStdio dr(fp);
  return load_param(dr);
}

#if NCNN_STRING
int Net::load_param(const ncnn::DataReader &dr) {
#define SCAN_VALUE(fmt, v)                \
    if (dr.scan(fmt, &v) != 1)            \
    {                                     \
        NCNN_LOGE("parse " #v " failed"); \
        return -1;                        \
    }

  int magic = 0;
  SCAN_VALUE("%d", magic)
  // 开头关键字
  if (magic != 7767517) {
    NCNN_LOGE("param is too old, please regenerate");   // 看一蛤
    return -1;
  }

  // 接受前2个字
  int layer_count = 0;
  int blob_count = 0;
  SCAN_VALUE("%d", layer_count)
  SCAN_VALUE("%d", blob_count)
  if (layer_count <= 0 || blob_count <= 0) {
    NCNN_LOGE("invalid layer_count or blob_count");
    return -1;
  }
  d_->layers.resize((size_t) layer_count);    // 分配layer
  d_->blobs.resize((size_t) blob_count);    // 分配blob

  // 省略了NCNN_VULKAN相关内容
  ncnn_M::ParamDict pd;  // int ncnn::ParamDict::load_param(const ncnn::DataReader&) is protected within this context

  int blob_index = 0;
  for (int i = 0; i < layer_count; i++) {
    char layer_type[256];
    char layer_name[256];
    int bottom_count = 0;
    int top_count = 0;
    SCAN_VALUE("%255s", layer_type)
    SCAN_VALUE("%255s", layer_name)
    SCAN_VALUE("%d", bottom_count)
    SCAN_VALUE("%d", top_count)

    ncnn::Layer *layer = ncnn::create_layer(layer_type);
    if (!layer) {
      layer = create_custom_layer(layer_type);
    }
    if (!layer) {
      NCNN_LOGE("layer %s not exists or registered", layer_type);
      clear();
      return -1;
    }
    /*
     if (layer->use_int8_inference) {
     // no int8 gpu or packing layout support yet
     opt.use_vulkan_compute = false;
     opt.use_packing_layout = false;
     opt.use_fp16_storage = false;
     opt.use_bf16_storage = false;
     }

     #if NCNN_VULKAN
     if (opt.use_vulkan_compute)
     layer->vkdev = d->vkdev;
     #endif // NCNN_VULKAN
     */

    layer->type = std::string(layer_type);
    layer->name = std::string(layer_name);
    LOG(INFO) << "## new layer[" << i << "] name: " << layer->name << " type: "
              << layer->type;

    // 层输出赋值
    layer->bottoms.resize(bottom_count);

    // 大概知道问题了，ncnn_M::Net.d和ncnn::Net.d不是一个东西，导致ncnn::Net.d需要的东东访问不了
    for (int j = 0; j < bottom_count; j++)  // 依次访问该层输出
        {
      char bottom_name[256];
      SCAN_VALUE("%255s", bottom_name)

//          for (size_t i = 0; i < d->blobs.size(); i++) {
//            const ncnn::Blob& blob = d->blobs[i];
//            std::cout << "blob[" << i << "] = " << blob.name << std::endl;
//          }
//          return 0;

      int bottom_blob_index = find_blob_index_by_name_MICRO(bottom_name);  // 从net.d.blobs中查找
      if (bottom_blob_index == -1)  // 如果不在net.d.blobs里面，则将下一个创建为该blob
          {
        ncnn::Blob &blob = d_->blobs[blob_index];  // blob变量绑定为下一个blob_index

        bottom_blob_index = blob_index;

        blob.name = std::string(bottom_name);  // 名字赋值
        LOG(INFO) << "### new blob " << blob.name;

        blob_index++;
      }

      ncnn::Blob &blob = d_->blobs[bottom_blob_index];

      blob.consumer = i;  // 赋值blob.consumer的关系

      layer->bottoms[j] = bottom_blob_index;  // 赋值layer->bottoms
    }

    // 层输入赋值
    layer->tops.resize(top_count);
    for (int j = 0; j < top_count; j++) {
      ncnn::Blob &blob = d_->blobs[blob_index];  // 一定会分配下一个blob，作为输入

      char blob_name[256];
      SCAN_VALUE("%255s", blob_name)

      blob.name = std::string(blob_name);
      //             NCNN_LOGE("new blob %s", blob_name);

      blob.producer = i;  // 赋值blob.producer的关系

      layer->tops[j] = blob_index;  // 赋值layer->tops

      blob_index++;
    }

    // layer specific params
    int pdlr = pd.load_param(dr);
    if (pdlr != 0) {
      NCNN_LOGE("**** ParamDict load_param failed");
      continue;
    }

    // pull out top shape hints
    ncnn::Mat shape_hints = pd.get(30, ncnn::Mat());
    LOG(INFO) << "     shape_hints: " << shape_hints.empty();
    if (!shape_hints.empty()) {
      const int *psh = shape_hints;
      for (int j = 0; j < top_count; j++) {
        ncnn::Blob &blob = d_->blobs[layer->tops[j]];

        int dims = psh[0];
        if (dims == 1) {
          blob.shape = ncnn::Mat(psh[1], (void*) 0, 4u, 1);
          LOG(INFO) << "     shape_hints[]: " << psh[1];
        }
        if (dims == 2) {
          blob.shape = ncnn::Mat(psh[1], psh[2], (void*) 0, 4u, 1);
          LOG(INFO) << "     shape_hints[]: " << psh[1] << ", " << psh[2];
        }
        if (dims == 3) {
          blob.shape = ncnn::Mat(psh[1], psh[2], psh[3], (void*) 0, 4u, 1);
          LOG(INFO) << "     shape_hints[]: " << psh[1] << ", " << psh[2]
                    << ", " << psh[3];
        }

        psh += 4;
      }
    }

    // set bottom and top shape hints
    layer->bottom_shapes.resize(bottom_count);
    for (int j = 0; j < bottom_count; j++) {
      layer->bottom_shapes[j] = d_->blobs[layer->bottoms[j]].shape;  // 调整该层输入blob的大小
    }

    layer->top_shapes.resize(top_count);
    for (int j = 0; j < top_count; j++) {
      layer->top_shapes[j] = d_->blobs[layer->tops[j]].shape;  // 调整该层输出blob的大小
    }

    // int lr = layer->load_param(pd);  // 还得做一个将ncnn_M::ParamDict -> ncnn::ParamDict的工具
    int lr = layer->load_param(*((ncnn::ParamDict*) (&pd)));  // 还得做一个将ncnn_M::ParamDict -> ncnn::ParamDict的工具，不过目前还来强行cast还是稳定的
    if (lr != 0) {
      NCNN_LOGE("layer load_param failed");
      continue;
    }

    d_->layers[i] = layer;
  }

#undef SCAN_VALUE

  return 0;
}
#endif

int Net::custom_layer_to_index(const char *type) {
  return 0;
}
ncnn::Layer* Net::create_custom_layer(const char *type) {
  return nullptr;
}
ncnn::Layer* Net::create_custom_layer(int index) {
  return nullptr;
}

int Net::load_model(const char *modelpath) {
  FILE *fp = fopen(modelpath, "rb");
  if (!fp) {
    NCNN_LOGE("fopen %s failed", modelpath);
    return -1;
  }

  int ret = load_model(fp);
  fclose(fp);
  return ret;
}

int Net::load_model(FILE *fp) {
  ncnn::DataReaderFromStdio dr(fp);
  return load_model(dr);
}

int Net::load_model(const ncnn::DataReader &dr) {
  if (d_->layers.empty()) {
    NCNN_LOGE("network graph not ready");
    return -1;
  }

  // load file
  int ret = 0;

  ncnn_M::ModelBinFromDataReader mb(dr);
  for (size_t i = 0; i < d_->layers.size(); i++) {  // browse all layers
    ncnn::Layer *layer = d_->layers[i];

    std::ofstream ofile(raw_model_profile_path, std::ios::out | std::ios::app);
    ofile << "Layer " << layer->type << "(" << layer->name << ")\n";
    ofile.close();

    //Here we found inconsistent content in the parameter file.
    if (!layer) {
      NCNN_LOGE(
          "load_model error at layer %d, parameter file has inconsistent content.",
          (int )i);
      ret = -1;
      break;
    }

    int lret = layer->load_model(mb);  // 从文件中读如
    if (lret != 0) {
      NCNN_LOGE("layer load_model %d failed", (int )i);
      ret = -1;
      break;
    }
  }

  // d_->fuse_network();  // 里面有requant相关的东东，但是我用不上

  for (size_t i = 0; i < d_->layers.size(); i++) {
    ncnn::Layer *layer = d_->layers[i];

    //Here we found inconsistent content in the parameter file.
    if (!layer) {
      NCNN_LOGE(
          "load_model error at layer %d, parameter file has inconsistent content.",
          (int )i);
      ret = -1;
      break;
    }

    ncnn::Option opt1 = opt;

    int cret = layer->create_pipeline(opt1);
    if (cret != 0) {
      NCNN_LOGE("layer create_pipeline %d failed", (int )i);
      ret = -1;
      break;
    }
  }

  return ret;
}

}  // namespace ncnn_M

int test(const char *out_dir) {
  GLogHelper gh("");

  // ensure out_dir is brandly new
  namespace fs = std::filesystem;
  if (fs::exists(fs::path(out_dir))) {
    std::cout << out_dir << " exists! \n";
    fs::remove_all(fs::path(out_dir));
  }
  fs::create_directory(fs::path(out_dir));
  // put raw_profile_path in out_dir
  raw_model_profile_path = (fs::path(out_dir) / fs::path(raw_model_profile_path)).string();

  ncnn_M::Net net;
  int ret;

  ret = net.load_param("../data/model/mobilenetv2_f32/mobilenet_v2_f32.param");
  std::cout << "net.load_param() -> " << ret << std::endl;

  ret = net.load_model("../data/model/mobilenetv2_f32/mobilenet_v2_f32.bin");
  std::cout << "net.load_model() -> " << ret << std::endl;

  // std::cout << "__cplusplus: " << __cplusplus << std::endl;
  return EXIT_SUCCESS;
}

int main() {
  const char out_dir[] = "ncnn_micro_out";
  int r = test(out_dir);

  return r;
}
