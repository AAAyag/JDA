#include <ctime>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/cart.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"
#include <map>

using namespace cv;
using namespace std;

namespace jda {

static int YO = 0;

JoinCascador::JoinCascador() {
  const Config& c = Config::GetInstance();
  T = c.T;
  K = c.K;
  landmark_n = c.landmark_n;
  tree_depth = c.tree_depth;
  current_stage_idx = 0;
  current_cart_idx = -1;
  btcarts.reserve(T);
  for (int t = 0; t < T; t++) {
    btcarts.push_back(BoostCart(t));
  }
}
JoinCascador::~JoinCascador() {
}

void JoinCascador::Train(DataSet& pos, DataSet& neg) {
  this->pos = &pos;
  this->neg = &neg;
  const int start_of_stage = current_stage_idx;
  for (int t = start_of_stage; t < T; t++) {
    current_stage_idx = t;
    if (current_stage_idx != start_of_stage) {
      current_cart_idx = -1;
    }
    LOG("Train %d th stages", t + 1);
    TIMER_BEGIN
      btcarts[t].Train(pos, neg);
      LOG("End of train %d th stages, costs %.4lf s", t + 1, TIMER_NOW);
    TIMER_END
    LOG("Snapshot current Training Status");
    // snapshot
    DataSet::Snapshot(pos, neg);
    Snapshot();
  }
}

void JoinCascador::Snapshot() const {
  int stage_idx = current_stage_idx;
  int cart_idx = current_cart_idx;
  char buff1[256];
  char buff2[256];
  time_t t = time(NULL);
  strftime(buff1, sizeof(buff1), "%Y%m%d-%H%M%S", localtime(&t));
  sprintf(buff2, "../model/jda_tmp_%s_stage_%d_cart_%d.model", \
          buff1, stage_idx + 1, cart_idx + 1);

  FILE* fd = fopen(buff2, "wb");
  JDA_Assert(fd, "Can not open a temp file to save the model");

  SerializeTo(fd);

  fclose(fd);
}

void JoinCascador::Resume(FILE* fd) {
  SerializeFrom(fd);
}

void JoinCascador::SerializeTo(FILE* fd) const {
  fwrite(&YO, sizeof(YO), 1, fd);
  fwrite(&T, sizeof(int), 1, fd); // number of stages
  fwrite(&K, sizeof(int), 1, fd); // number of trees per stage
  fwrite(&landmark_n, sizeof(int), 1, fd); // number of landmarks
  fwrite(&tree_depth, sizeof(int), 1, fd); // tree depth
  // \note
  //  current_stage_idx and current_cart_idx
  //  these two variable are setted to indicate the training status of JDA
  //  current_stage_idx = 0, current_cart_idx = 4    --> jda_xxxx_stage_1_cart_5.model
  //  current_stage_idx = 1, current_catr_idx = K-1  --> jda_xxxx_stage_2_cart_K.model
  //  whene current_cart_idx == K-1, we assume that the global regression of this stage
  //  is already done and these two variable will be saved as `current_stage_idx+1` and
  //  `current_cart_idx = -1`
  if (current_cart_idx == K - 1) {
    int idx = current_stage_idx + 1;
    fwrite(&idx, sizeof(int), 1, fd);
    idx = -1;
    fwrite(&idx, sizeof(int), 1, fd);
  }
  else {
    int idx = current_stage_idx;
    fwrite(&idx, sizeof(int), 1, fd);
    idx = current_cart_idx;
    fwrite(&idx, sizeof(int), 1, fd);
  }
  // mean shape
  fwrite(mean_shape.ptr<double>(0), sizeof(double), mean_shape.cols, fd);
  // btcarts
  for (int t = 0; t < T; t++) {
    const BoostCart& btcart = btcarts[t];
    for (int k = 0; k < K; k++) {
      const Cart& cart = btcart.carts[k];
      cart.SerializeTo(fd);
    }
    // global regression parameters
    const double* w_ptr;
    const int rows = btcart.w.rows;
    const int cols = btcart.w.cols;
    for (int i = 0; i < rows; i++) {
      w_ptr = btcart.w.ptr<double>(i);
      fwrite(w_ptr, sizeof(double), cols, fd);
    }
  }
  fwrite(&YO, sizeof(YO), 1, fd);
}

void JoinCascador::SerializeFrom(FILE* fd) {
  int tmp;
  fread(&YO, sizeof(YO), 1, fd);
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == T, "T is wrong!");
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == K, "K is wrong!");
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == landmark_n, "landmark_n is wrong!");
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == tree_depth, "tree_depth is wrong!");
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(0 <= tmp && tmp <= T, "current_stage_idx out of range");
  current_stage_idx = tmp;
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(-1 <= tmp && tmp < K, "current_cart_idx out of range");
  current_cart_idx = tmp;

  // mean shape
  mean_shape.create(1, 2 * landmark_n);
  fread(mean_shape.ptr<double>(0), sizeof(double), mean_shape.cols, fd);

  for (int t = 0; t < T; t++) {
    BoostCart& btcart = btcarts[t];
    for (int k = 0; k < K; k++) {
      Cart& cart = btcart.carts[k];
      cart.SerializeFrom(fd);
    }
    // global regression parameters
    double* w_ptr;
    const int w_rows = K * (1 << (tree_depth - 1));
    const int w_cols = 2 * landmark_n;
    for (int i = 0; i < w_rows; i++) {
      w_ptr = btcart.w.ptr<double>(i);
      fread(w_ptr, sizeof(double), w_cols, fd);
    }
  }
  fread(&YO, sizeof(YO), 1, fd);
}

bool JoinCascador::Validate(const Mat& img, const Mat& img_h, const Mat& img_q, \
                            double& score, Mat_<double>& shape, int& n) const {
  const Config& c = Config::GetInstance();
  DataSet::RandomShape(mean_shape, shape);
  score = 0;
  n = 0;
  Mat_<int> lbf(1, c.K);
  int* lbf_ptr = lbf.ptr<int>(0);
  const int base = 1 << (c.tree_depth - 1);
  int offset = 0;
  // stage [0, current_stage_idx)
  for (int t = 0; t < current_stage_idx; t++) {
    const BoostCart& btcart = btcarts[t];
    offset = 0;
    for (int k = 0; k < c.K; k++) {
      const Cart& cart = btcart.carts[k];
      int idx = cart.Forward(img, img_h, img_q, shape);
      score += cart.scores[idx];
      n++;
      if (score < cart.th) {
        // not a face
        return false;
      }
      lbf_ptr[k] = offset + idx;
      offset += base;
    }
    // global regression
    shape += btcart.GenDeltaShape(lbf);
  }
  // current stage, cart [0, current_cart_idx]
  for (int k = 0; k <= current_cart_idx; k++) {
    const Cart& cart = btcarts[current_stage_idx].carts[k];
    int idx = cart.Forward(img, img_h, img_q, shape);
    score += cart.scores[idx];
    n++;
    if (score < cart.th) {
      // not a face
      return false;
    }
  }
  return true;
}

/*!
 * \breif detect single scale
 */
static void detectSingleScale(const JoinCascador& joincascador, const Mat& img, int step, \
                              int win_size, vector<Rect>& rects, vector<double>& scores, \
                              vector<Mat_<double> >& shapes, DetectionStatisic& statisic) {
  const Config& c = Config::GetInstance();
  const int win_w = win_size;
  const int win_h = win_size;
  const int x_max = img.cols - win_w;
  const int y_max = img.rows - win_h;
  const int x_step = step;
  const int y_step = step;
  int x = 0;
  int y = 0;

  rects.clear();
  scores.clear();
  shapes.clear();

  Mat patch; // used as a patch
  Mat patch_h;
  Mat patch_q;

  while (y <= y_max) {
    while (x <= x_max) {
      Rect roi(x, y, win_w, win_h);
      double score;
      Mat_<double> shape;
      int reject_length;
      cv::resize(img(roi), patch, Size(c.img_o_size, c.img_o_size));
      cv::resize(img(roi), patch_h, Size(c.img_h_size, c.img_h_size));
      cv::resize(img(roi), patch_q, Size(c.img_q_size, c.img_q_size));
      bool is_face = joincascador.Validate(patch, patch_h, patch_q, score, shape, reject_length);
      if (is_face) {
        rects.push_back(roi);
        scores.push_back(score);
        shapes.push_back(shape);
        statisic.face_patch_n++;
      }
      else {
        statisic.nonface_patch_n++;
        statisic.cart_gothrough_n += reject_length;
      }
      x += x_step;
    }
    x = 0;
    y += y_step;
  }
}

/*!
 * \breif detect multi scale
 * \note    detection parameters can be configured in `config.json`
 */
static void detectMultiScale(const JoinCascador& joincascador, const Mat& img, \
                             vector<Rect>& rects, vector<double>& scores, \
                             vector<Mat_<double> >& shapes, DetectionStatisic& statisic) {
  const Config& c = Config::GetInstance();
  const int win_w = c.img_o_size;
  const int win_h = c.img_o_size;
  int width = img.cols;
  int height = img.rows;
  const double factor = c.fddb_scale_factor;
  double scale = 1.;
  const int step = c.fddb_step;
  Mat img_ = img.clone();

  rects.clear();
  scores.clear();
  shapes.clear();

  while ((width >= win_w) && (height >= win_h)) {
    vector<Rect> rects_;
    vector<double> scores_;
    vector<Mat_<double> > shapes_;
    detectSingleScale(joincascador, img_, step, win_w, rects_, scores_, shapes_, statisic);
    const int n = rects_.size();
    for (int i = 0; i < n; i++) {
      Rect& r = rects_[i];
      r.x *= scale; r.y *= scale;
      r.width *= scale; r.height *= scale;
    }
    rects.insert(rects.end(), rects_.begin(), rects_.end());
    scores.insert(scores.end(), scores_.begin(), scores_.end());
    shapes.insert(shapes.end(), shapes_.begin(), shapes_.end());

    scale *= factor;
    width = int(width / factor);
    height = int(height / factor);
    cv::resize(img_, img_, Size(width, height));
  }
  // statisic
  statisic.patch_n = statisic.face_patch_n + statisic.nonface_patch_n;
  statisic.average_cart_n = statisic.cart_gothrough_n / statisic.nonface_patch_n;
}

static void detectMultiScale1(const JoinCascador& joincascador, const Mat& img, \
                              vector<Rect>& rects, vector<double>& scores, \
                              vector<Mat_<double> >& shapes, DetectionStatisic& statisic) {
  const Config& c = Config::GetInstance();
  int win_w = c.fddb_minimum_size;
  int win_h = c.fddb_minimum_size;
  int step_size = c.fddb_step;
  const double factor = c.fddb_scale_factor;
  Mat img_o, img_h, img_q;
  int img_o_w, img_o_h;
  int img_h_w, img_h_h;
  int img_q_w, img_q_h;
  img_o_w = img.cols;
  img_o_h = img.rows;
  img_h_w = int(img.cols / std::sqrt(2.));
  img_h_h = int(img.rows / std::sqrt(2.));
  img_q_w = img.cols / 2;
  img_q_h = img.rows / 2;

  img_o = img.clone();
  cv::resize(img, img_h, Size(img_h_w, img_h_h));
  cv::resize(img, img_q, Size(img_q_w, img_q_h));

  while (win_w <= img_o_w && win_h <= img_o_h) {
    int x, y, x_max, y_max;
    x = y = 0;
    x_max = img_o_w - win_w;
    y_max = img_o_h - win_h;
    while (y <= y_max) {
      while (x <= x_max) {
        Rect roi_o(x, y, win_w, win_h);
        double r = std::sqrt(2.);
        Rect roi_h(int(x / r), int(y / r), int(win_w / r), int(win_h / r));
        Rect roi_q(x / 2, y / 2, win_w / 2, win_h / 2);
        //roi_h.width = std::min(roi_h.width, img_h.cols - roi_h.x);
        //roi_h.height = std::min(roi_h.height, img_h.rows - roi_h.y);
        //roi_q.width = std::min(roi_q.width, img_q.cols - roi_q.x);
        //roi_q.height = std::min(roi_q.height, img_q.rows - roi_q.y);
        double score;
        Mat_<double> shape;
        int reject_length;
        Mat patch_o = img_o(roi_o);
        Mat patch_h = img_h(roi_h);
        Mat patch_q = img_q(roi_q);
        bool is_face = joincascador.Validate(patch_o, patch_h, patch_q, score, shape, reject_length);
        if (is_face) {
          rects.push_back(roi_o);
          scores.push_back(score);
          shapes.push_back(shape);
          statisic.face_patch_n++;
        }
        else {
          statisic.nonface_patch_n++;
          statisic.cart_gothrough_n += reject_length;
        }
        x += step_size;
      }
      x = 0;
      y += step_size;
    }
    win_w = int(win_w*factor);
    win_h = int(win_h*factor);
  }
  // statisic
  statisic.patch_n = statisic.face_patch_n + statisic.nonface_patch_n;
  statisic.average_cart_n = statisic.cart_gothrough_n / statisic.nonface_patch_n;
}

/*!
 * \breif nms Non-maximum suppression
 *  the algorithm is from https://github.com/ShaoqingRen/SPP_net/blob/master/nms%2Fnms_mex.cpp
 *
 * \param rects     area of faces
 * \param scores    score of faces
 * \param overlap   overlap threshold
 * \return          picked index
 */
/*
static vector<int> nms(const vector<Rect>& rects, const vector<double>& scores, \
                       vector<int>& Srect, double overlap) {
  const int n = rects.size();
  vector<double> areas(n);

  typedef std::multimap<double, int> ScoreMapper;
  ScoreMapper map;
  for (int i = 0; i < n; i++) {
    map.insert(ScoreMapper::value_type(scores[i], i));
    areas[i] = rects[i].width*rects[i].height;
  }

  int picked_n = 0;
  vector<int> picked(n);
  while (map.size() != 0) {
    int last = map.rbegin()->second; // get the index of maximum score value
    picked[picked_n] = last;
    picked_n++;

    for (ScoreMapper::iterator it = map.begin(); it != map.end();) {
      int idx = it->second;
      double x1 = std::max(rects[idx].x, rects[last].x);
      double y1 = std::max(rects[idx].y, rects[last].y);
      double x2 = std::min(rects[idx].x + rects[idx].width, rects[last].x + rects[last].width);
      double y2 = std::min(rects[idx].y + rects[idx].height, rects[last].y + rects[last].height);
      double w = std::max(0., x2 - x1);
      double h = std::max(0., y2 - y1);
      double ov = w*h / (areas[idx] + areas[last] - w*h);
      if (ov > overlap) {
        ScoreMapper::iterator tmp = it;
        tmp++;
        map.erase(it);
        it = tmp;
      }
      else{
        it++;
      }
    }
  }

  picked.resize(picked_n);
  return picked;
}
*/
vector<double> Logistic(vector<double> scores ,vector<int> index);
int Partation(Mat predicate,vector<int>& label);
int Find(vector<int>& parent,int x);

vector<int> nms(vector<Rect>& rects, vector<double>& scores, vector<int>& Srect, double overlap, Mat Img, vector<Mat_<double> >& shapes) {
  int numCandidates = rects.size();
  Mat predicate = Mat::eye(numCandidates,numCandidates,IPL_DEPTH_1U);
  for(int i = 0;i<numCandidates;i++){
    for(int j = i+1;j<numCandidates;j++){
      int h = min(rects[i].y+rects[i].height,rects[j].y+rects[j].height) - max(rects[i].y,rects[j].y);
      int w = min(rects[i].x+rects[i].width,rects[j].x+rects[j].width) - max(rects[i].x,rects[j].x);
      int s = max(h,0)*max(w,0);

      if ((double)s/(double)(rects[i].width*rects[i].height+rects[j].width*rects[j].height-s)>=overlap){
        predicate.at<bool>(i,j) = 1;
        predicate.at<bool>(j,i) = 1;
      }
    }
  }

  vector<int> label;

  int numLabels = Partation(predicate,label);

  vector<Rect> Rects;
  vector<Mat_<double> > Shapes;
  Srect.resize(numLabels);
  vector<int> neighbors;
  neighbors.resize(numLabels);
  vector<double> Score;
  Score.resize(numLabels);

  for(int i = 0;i<numLabels;i++){
    vector<int> index;
    for(int j = 0;j<numCandidates;j++){
      if(label[j]==i)
        index.push_back(j);
    }
    vector<double> weight;
    weight = Logistic(scores,index);
    double sumScore=0;
    for(int j=0;j<weight.size();j++)
      sumScore+=weight[j];
    Score[i] = sumScore;
    neighbors[i]=index.size();

    if (sumScore == 0){
      for(int j=0;j<weight.size();j++)
        weight[j] = 1/sumScore;
    }
    else{
      for(int j=0;j<weight.size();j++)
        weight[j] = weight[j]/sumScore;
    }
    double size = 0;
    double col = 0;
    double row = 0;
    for(int j=0;j<index.size();j++){
      size += rects[index[j]].width*weight[j];
    }
    Srect[i] = (int)floor(size);
    for(int j=0;j<index.size();j++){
      col += (rects[index[j]].x + rects[index[j]].width/2)*weight[j];
      row += (rects[index[j]].y + rects[index[j]].width/2)*weight[j];
    }
    const int landmark_n = shapes[0].cols / 2;
    Mat_<double> shape = Mat_<double>::zeros(1,2*landmark_n);
    for (int j = 0; j < landmark_n; j++) {
      for(int l = 0; l < index.size(); l++){
        shape(0, 2 * j) += shapes[index[l]](0, 2 * j)*weight[l];
        shape(0, 2 * j + 1) += shapes[index[l]](0, 2 * j + 1)*weight[l];
      }
    }
    int x = floor(col-size/2);
    int y = floor(row-size/2);
    Rect roi(x,y,Srect[i],Srect[i]);
    Shapes.push_back(shape);
    Rects.push_back(roi);
  }


  predicate = Mat::zeros(numLabels,numLabels,IPL_DEPTH_1U);

  for(int i = 0;i<numLabels;i++){
    for(int j = i+1;j<numLabels;j++){
      int h = min(Rects[i].y+Rects[i].height,Rects[j].y+Rects[j].height) - max(Rects[i].y,Rects[j].y);
      int w = min(Rects[i].x+Rects[i].width,Rects[j].x+Rects[j].width) - max(Rects[i].x,Rects[j].x);
      int s = max(h,0)*max(w,0);

      if((double)s/(double)(Rects[i].width*Rects[i].height)>=overlap || (double)s/(double)(Rects[j].width*Rects[j].height)>=overlap)
      {
        predicate.at<bool>(i,j) = 1;
        predicate.at<bool>(j,i) = 1;
      }
    }
  }

  vector<int> flag;
  flag.resize(numLabels);
  for(int i = 0;i<numLabels;i++)
    flag[i]=1;

  for(int i = 0;i<numLabels;i++){
    vector<int> index;
    for(int j = 0;j<numLabels;j++){
      if(predicate.at<bool>(j,i)==1)
        index.push_back(j);
    }
    if(index.size()==0)
      continue;

    double s = 0;
    for(int j  = 0;j<index.size();j++){
      if(Score[index[j]]>s)
        s = Score[index[j]];
    }
    if(s>Score[i])
      flag[i]=0;
  }

  vector<int> picked;
  for(int i = 0;i<numLabels;i++){
    if(flag[i]){
      picked.push_back(i);
    }
  }

  int height = Img.rows;
  int width = Img.cols;

  for(int i = 0;i<picked.size();i++){
    int idx = picked[i];
    if(Rects[idx].x<0)
      Rects[idx].x = 0;

    if(Rects[idx].y<0)
      Rects[idx].y = 0;

    if(Rects[idx].y+Rects[idx].height>height)
      Rects[idx].height = height-Rects[idx].y;

    if(Rects[idx].x+Rects[idx].width>width)
      Rects[idx].width= width-Rects[idx].x;
  }
  rects = Rects;
  shapes = Shapes;
  scores = Score;
  return picked;
}

vector<double> Logistic(vector<double> scores ,vector<int> index){
  vector<double> Y;
  for(int i = 0;i<index.size();i++){
    double tmp_Y = log(1+exp(scores[index[i]]));
    if(isinf(tmp_Y))
      tmp_Y = scores[index[i]];
    Y.push_back(tmp_Y);
  }
  return Y;
}

int Partation(Mat predicate,vector<int>& label){
  int N = predicate.cols;
  vector<int> parent;
  vector<int> rank;
  for(int i=0;i<N;i++){
    parent.push_back(i);
    rank.push_back(0);
  }

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if (predicate.at<bool>(i,j)==0)
        continue;
      int root_i = Find(parent,i);
      int root_j = Find(parent,j);

      if(root_j != root_i){
        if (rank[root_j] < rank[root_i])
          parent[root_j] = root_i;
        else if (rank[root_j] > rank[root_i])
          parent[root_i] = root_j;
        else{
          parent[root_j] = root_i;
          rank[root_i] = rank[root_i] + 1;
        }
      }
    }
  }

  int nGroups = 0;
  label.resize(N);
  for(int i=0;i<N;i++){
    if(parent[i]==i){
      label[i] = nGroups;
      nGroups++;
    }
    else label[i] = -1;
  }

  for(int i=0;i<N;i++){
    if(parent[i]==i)
      continue;
    int root_i = Find(parent,i);
    label[i]=label[root_i];
  }

  return nGroups;
}

int Find(vector<int>& parent,int x){
  int root = parent[x];
  if(root != x)
    root = Find(parent,root);
  return root;
}

int JoinCascador::Detect(const Mat& img, vector<Rect>& rects, vector<double>& scores, \
                         vector<Mat_<double> >& shapes, DetectionStatisic& statisic) const {
  const Config& c = Config::GetInstance();
  vector<Rect> rects_;
  vector<double> scores_;
  vector<Mat_<double> > shapes_;

  if (c.fddb_detect_method == 0) {
    detectMultiScale(*this, img, rects_, scores_, shapes_, statisic);
  }
  else {
    detectMultiScale1(*this, img, rects_, scores_, shapes_, statisic);
  }
  
  //const double overlap = 0.3;
  const double overlap = c.fddb_overlap;
  vector<int> picked;
  vector<int> Srect;
  if (c.fddb_nms) {
    picked = nms(rects_, scores_, Srect, overlap, img, shapes_);
  }
  else {
    const int n = rects_.size();
    for(int i = 0;i<n;i++)
      Srect.push_back(rects_[i].width);
    picked.resize(n);
    for (int i = 0; i < n; i++) picked[i] = i;
  }

  const int n = picked.size();
  rects.resize(n);
  scores.resize(n);
  shapes.resize(n);

  // relocate the shape points
  for (int i = 0; i < n; i++) {
    const int index = picked[i];
    const Rect& rect = rects_[index];
    Mat_<double>& shape = shapes_[index];
    const int landmark_n = shape.cols / 2;
    for (int j = 0; j < landmark_n; j++) {
      shape(0, 2 * j) = rect.x + shape(0, 2 * j)*rect.width;
      shape(0, 2 * j + 1) = rect.y + shape(0, 2 * j + 1)*rect.height;
    }
    rects[i] = rect;
    shapes[i] = shape;
    scores[i] = scores_[index];
  }

  int imgWidth = img.cols;
  int imgHeight = img.rows;

  for(int i = 0;i<picked.size();i++){
    int idx = picked[i];
    int delta = floor(Srect[idx]*0.1);
    int y0 = max(rects_[idx].y - floor(3.0 * delta),0.);
    int y1 = min(rects_[idx].y + Srect[idx],imgHeight);
    int x0 = max(rects_[idx].x + floor(0.25 * delta),0.);
    int x1 = min(rects_[idx].x + Srect[idx] - floor(0.25 * delta),(double)imgWidth);

    rects_[idx].y = y0;
    rects_[idx].x = x0;
    rects_[idx].width = x1-x0 + 1;
    rects_[idx].height = y1-y0 + 1;
  }

  return n;
}

} // namespace jda
