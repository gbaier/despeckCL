#ifndef BBOX_H
#define BBOX_H

class bbox {
    public:
        int h_low;
        int h_up;
        int w_low;
        int w_up;

        bbox(const int h_low, const int w_low, const int h_up, const int w_up);

        bool valid(const int height, const int width);
};

#endif
