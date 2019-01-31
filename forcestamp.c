// #include </Users/hanc/.pyenv/versions/3.7.1/include/python3.7m//Python.h>
#include "C:\Users\hanc\AppData\Local\Programs\Python\Python37\include\Python.h"
// #include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// #define PI acos(-1)

typedef struct {
    double x;
    double y;
} COORD;

typedef struct {
    COORD *co;
    unsigned int len;
} COORD_ARRAY;

typedef struct {
    COORD_ARRAY *co_array;
    unsigned int len;
} SEQ_COORD_ARRAY;

double _distance(COORD *a, COORD *b) {
    return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
}

double _minimum(double a, double b) {
    double c;
    if (a >= b) {
        c = b;
    } else {
        c = a;
    }
    return c;
}

void _allocateMemArray(COORD_ARRAY *c) {
    c->co = (COORD *)malloc(sizeof(COORD) * c->len);
    if (c->co == NULL) {
        printf("memory allocation error!");
        exit(-1);
    }
}

void _allocateMemSeqArray(SEQ_COORD_ARRAY *c) {
    c->co_array = (COORD_ARRAY *)malloc(sizeof(COORD_ARRAY) * c->len);
    if (c->co_array == NULL) {
        printf("memory allocation error!");
        exit(-1);
    }
}

void _findCenter(
    COORD_ARRAY centers,
    COORD_ARRAY peaks, 
    COORD_ARRAY *markerCenters,
    double markerRadius,
    double distanceTolerance
    ) {
    // declaration
    unsigned int distanceCount;
    unsigned int inboundPeakCount;
    unsigned int i, j, k;

    // filter out marker centers which has given radius
    k = 0;
    for (i = 0; i < centers.len; i++) {
        distanceCount = 0;
        inboundPeakCount = 0;
        for (j = 0; j < peaks.len; j++) {
            // printf("i: %d, j: %d\n", i, j);
            // printf("cnt: %f, %f, peak: %f, %f\n", (centers.co + i)->x, (centers.co + i)->y,(peaks.co + j)->x, (peaks.co + j)->y);
            // printf("distance: %f\n", _distance(centers.co + i, peaks.co + j));
            if (_distance(centers.co + i, peaks.co + j) < markerRadius + distanceTolerance &&
                _distance(centers.co + i, peaks.co + j) >= markerRadius - distanceTolerance) {
                distanceCount++;
            }
            if (_distance(centers.co + i, peaks.co + j) < markerRadius - distanceTolerance) {
                inboundPeakCount++;
            }
        }
        // printf("distance count: %d\n", distanceCount);
        if (distanceCount > 4 && inboundPeakCount < 4) {
            // printf("found point!\n");
            // printf("k: %d\n", k);
            // allocate more memory to mc
            // markerCenters->co = realloc(markerCenters->co, sizeof(COORD) * (k + 1));
            COORD *tmp = realloc(markerCenters->co, sizeof(COORD) * (k + 1));
            if (tmp == NULL) {
                printf("mcenters co allocation error!");
                free(markerCenters->co);
                exit(-1);
            } else {
                markerCenters->co = tmp;
            }
            (markerCenters->co + k)->x = (centers.co + i)->x;
            (markerCenters->co + k)->y = (centers.co + i)->y;
            k++;
            markerCenters->len = k;
            // printf("marker centers: x: %f, y: %f\n", (centers.co + i)->x, (centers.co + i)->y);
            // printf("marker center len: %d\n", k);
        }
    }
    // free(markerCenters->co);
}

void _clusterCenters(COORD_ARRAY c_in, SEQ_COORD_ARRAY *c_out, double markerRadius){
    unsigned int currentCluster;
    unsigned int cnt, i;
    double minDist, dist;
    bool isNew;

    // printf("start clustering center coords\n");
    // printf("c_in->len: %d\n", c_in.len);
    for (cnt = 0; cnt < c_in.len; cnt++) {
        if (c_out->len == 0) {
            // allocate memory
            c_out->len = 1;
            _allocateMemSeqArray(c_out);
            c_out->co_array->len = 1;
            // printf("c_out->co_array->len = %d\n", c_out->co_array->len); 
            _allocateMemArray(c_out->co_array);
            c_out->co_array->co->x = (c_in.co + cnt)->x;
            c_out->co_array->co->y = (c_in.co + cnt)->y;
            // numCluster++;
            // c_out->len++;
            // printf("c_out->len: %d\n", c_out->len);
            // printf("c_out->co_array->co->x: %f\n", c_out->co_array->co->x);
            // printf("c_out->co_array->co->y: %f\n", c_out->co_array->co->y);

        }
        else {
            currentCluster = 0;
            isNew = true;
            minDist = 1000.0;
            // printf("c_out->len in else: %d\n", c_out->len);
            for (i = 0; i < c_out->len; i++) {
                printf("i: %d\n", i);
                // check distance from existing cluster centers
                dist = _distance(c_in.co + cnt, (c_out->co_array + i)->co);
                // printf("distance: %f\n", dist);
                minDist = _minimum(minDist, dist);
                if (dist < markerRadius / 2) {
                    // add the center to current cluster
                    isNew = false;
                    currentCluster = i;
                    break;
                }
            }
            if (isNew) {
                // printf("it's new!\n");
                // if the point does not belong to any existing clusters
                // printf("min Dist: %f\n", minDist);
                if (minDist > markerRadius * 2.1) {
                    // add new cluster
                    // numCluster++;
                    c_out->len++;
                    // printf("c_out->len: %d\n", c_out->len);
                    // printf("(c_out->co_array + currentCluster)->len: %d\n", (c_out->co_array + currentCluster)->len);
                    COORD_ARRAY *tmp = realloc(c_out->co_array, sizeof(COORD_ARRAY) * c_out->len);
                    if (tmp == NULL) {
                        printf("memory allocation error");
                        free(c_out->co_array);
                        exit(-1);
                    } else {
                        c_out->co_array = tmp;
                    }
                    // printf("(c_out->co_array + currentCluster)->len: %d\n", (c_out->co_array + currentCluster)->len);
                    (c_out->co_array + c_out->len - 1)->len = 1;
                    _allocateMemArray(c_out->co_array + c_out->len - 1);
                    (c_out->co_array + c_out->len - 1)->co->x = (c_in.co + cnt)->x;
                    (c_out->co_array + c_out->len - 1)->co->y = (c_in.co + cnt)->y;
                    // printf("cluster %d\n", c_out->len - 1);
                    // printf("(c_out->co_array + c_out->len - 1)->co->x: %f\n", (c_out->co_array + c_out->len - 1)->co->x);
                    // printf("(c_out->co_array + c_out->len - 1)->co->y: %f\n", (c_out->co_array + c_out->len - 1)->co->y);
                }
            } else {
                // append to current cluster
                // printf("append to current cluster %d\n", currentCluster);
                // printf("(c_out->co_array + currentCluster)->len: %d\n", (c_out->co_array + currentCluster)->len);
                (c_out->co_array + currentCluster)->len++;
                // printf("(c_out->co_array + currentCluster)->len: %d\n", (c_out->co_array + currentCluster)->len);
                COORD *tmp = realloc((c_out->co_array + currentCluster)->co, sizeof(COORD) * (c_out->co_array + currentCluster)->len);
                if (tmp == NULL) {
                    printf("memory allocation error");
                    free((c_out->co_array + currentCluster)->co);
                    exit(-1);
                } else {
                    (c_out->co_array + currentCluster)->co = tmp;
                }

                ((c_out->co_array + currentCluster)->co + (c_out->co_array + currentCluster)->len - 1)->x = (c_in.co + cnt)->x;
                ((c_out->co_array + currentCluster)->co + (c_out->co_array + currentCluster)->len - 1)->y = (c_in.co + cnt)->y;
                // printf("x: %f, y: %f\n", (c_in.co + cnt)->x, (c_in.co + cnt)->y);
            }

        }
    }
}

void _getAccurateCenters(SEQ_COORD_ARRAY c_in, COORD_ARRAY *c_out) {
    unsigned int i, j;
    double averageCoordX, averageCoordY;
    // printf("starting center coords integration\n");
    // printf("c_in.len: %d\n", c_in.len);
    c_out->len = c_in.len;
    // printf("c_out->len: %d\n", c_out->len);
    _allocateMemArray(c_out);
    for (i = 0; i < c_in.len; i++) {
        averageCoordX = averageCoordY = 0;
        for (j = 0; j < (c_in.co_array + i)->len; j++) {

            averageCoordX += ((c_in.co_array + i)->co + j)->x;
            averageCoordY += ((c_in.co_array + i)->co + j)->y;
        }
        averageCoordX /= (c_in.co_array + i)->len;
        averageCoordY /= (c_in.co_array + i)->len;
        // printf("average X: %f, Y: %f\n", averageCoordX, averageCoordY);
        (c_out->co + i)->x = averageCoordX;
        (c_out->co + i)->y = averageCoordY;
    }
}

void _findMarkerCenters(COORD_ARRAY circleCenters, 
            COORD_ARRAY peaks, 
            COORD_ARRAY *mcenters_filtered,
            double markerRadius, 
            double distanceTolerance) {
    COORD_ARRAY mcenters;
    SEQ_COORD_ARRAY clustered_centers;
    mcenters.len = 1;
    clustered_centers.len = 0;
    
    _findCenter(
        circleCenters,
        peaks, 
        &mcenters,
        markerRadius,
        distanceTolerance
        );

    _clusterCenters(mcenters, &clustered_centers, markerRadius);
    
    _getAccurateCenters(clustered_centers, mcenters_filtered);

}

int main() {

    COORD_ARRAY centers;
    COORD_ARRAY peaks;
    COORD_ARRAY mcenters;
    SEQ_COORD_ARRAY clustered_centers;
    COORD_ARRAY mcenters_filtered;
    double markerRadius = 20;
    double distanceTolerance = 0.5;
    unsigned int i, j;

    centers.len = 7;
    peaks.len = 11;
    mcenters.len = 0;
    clustered_centers.len = 0;

    _allocateMemArray(&centers);
    _allocateMemArray(&peaks);
    // _allocateMemArray(&mcenters);

    (centers.co + 0)->x = 20.368355962743387;
    (centers.co + 0)->y = 73.91223730849559;
    (centers.co + 1)->x = 3.2883736745415124;
    (centers.co + 1)->y = 144.98733568970397;
    (centers.co + 2)->x = 20.71162632545849;
    (centers.co + 2)->y = 114.01266431029602;
    (centers.co + 3)->x = 62.39096026315619;
    (centers.co + 3)->y = 55.935169703949285;
    (centers.co + 4)->x = 83.60903973684381;
    (centers.co + 4)->y = 32.064830296050715;
    (centers.co + 5)->x = 83.93039029059602;
    (centers.co + 5)->y = 71.90662184615894;
    (centers.co + 6)->x = 98.06960970940398;
    (centers.co + 6)->y = 40.09337815384105;

    (peaks.co + 0)->x = 2;
    (peaks.co + 0)->y = 66;
    (peaks.co + 1)->x = 4;
    (peaks.co + 1)->y = 6;
    (peaks.co + 2)->x = 4;
    (peaks.co + 2)->y = 125;
    (peaks.co + 3)->x = 5;
    (peaks.co + 3)->y = 176;
    (peaks.co + 4)->x = 6;
    (peaks.co + 4)->y = 60;
    (peaks.co + 5)->x = 20;
    (peaks.co + 5)->y = 134;
    (peaks.co + 6)->x = 64;
    (peaks.co + 6)->y = 36;
    (peaks.co + 7)->x = 82;
    (peaks.co + 7)->y = 52;
    (peaks.co + 8)->x = 98;
    (peaks.co + 8)->y = 7;
    (peaks.co + 9)->x = 99;
    (peaks.co + 9)->y = 124;
    (peaks.co + 10)->x = 100;
    (peaks.co + 10)->y = 60;


// peaks = [(2, 66), (4, 6), (4, 125), (5, 176), (6, 60), (20, 134), (64, 36), (82, 52), (98, 7), (99, 124), (100, 60)]


    // for (i = 0; i < centers.len; i++) {
    //     (centers.co + i)->x = i * 50;
    //     (centers.co + i)->y = i * 50;
    // }

    // for (i = 0; i < peaks.len; i++) {
    //     (peaks.co + i)->x = i / 4 * 50 + 20;
    //     (peaks.co + i)->y = i / 4 * 50;
    // }

    // for (i = 9; i < centers.len; i++) {
    //     (centers.co + i)->x = 50.3;
    //     (centers.co + i)->y = 50.2;
    // }

    // for (i = 10; i < centers.len; i++) {
    //     (centers.co + i)->x = -0.2;
    //     (centers.co + i)->y = 0.1;
    // }
    // for (i = 11; i < centers.len; i++) {
    //     (centers.co + i)->x = -0.03;
    //     (centers.co + i)->y = 0.3;
    // }

    // _findMarkerCenters(
    //         centers, 
    //         peaks, 
    //         &mcenters_filtered,
    //         markerRadius, 
    //         distanceTolerance
    // );

    _findCenter(
    centers,
    peaks, 
    &mcenters,
    markerRadius,
    distanceTolerance
    );

    printf("marker center len: %d\n", mcenters.len);
    for (i = 0; i < mcenters.len; i++) {
        printf("center coord x: %f, %f \n", (mcenters.co + i)->x, (mcenters.co + i)->y);
    }

    _clusterCenters(mcenters, &clustered_centers, markerRadius);

    printf("clustered_centers.len: %d\n", clustered_centers.len);
    for (i = 0; i < clustered_centers.len; i++){
        printf("cluster %d\n", i);
        for (j = 0; j < (clustered_centers.co_array + i)->len; j++) {
            printf("array %d\n", j);
            printf("(%f, %f)\n", (clustered_centers.co_array[i].co + j)->x, (clustered_centers.co_array[i].co + j)->y);
        }
    }

    _getAccurateCenters(clustered_centers, &mcenters_filtered);

    for (i = 0; i < mcenters_filtered.len; i++) {
        printf("marker centers: (%f, %f)\n", (mcenters_filtered.co + i)->x, (mcenters_filtered.co + i)->y);
    }
}


static PyObject* findMarkerCenters(PyObject *self, PyObject *args)
{
    static PyObject *center_list, *peak_list;
    static PyObject *tuple_item, *temp_x, *temp_y;

    unsigned int center_length, peak_length, i;
    double markerRadius, distanceTolerance;
    static COORD_ARRAY centers, peaks, mcenters_filtered;
    static COORD_ARRAY mcenters;
    static SEQ_COORD_ARRAY clustered_centers;
    mcenters.len = 0;
    clustered_centers.len = 0;
    
    // check if the list is parsable
    if (!PyArg_ParseTuple(args, "OOdd", &center_list, &peak_list, &markerRadius, &distanceTolerance))
        return NULL;

    // get length of the lists
    center_length = PyObject_Length(center_list);
    if (center_length < 0)
        return NULL;
    peak_length = PyObject_Length(peak_list);
    if (peak_length < 0)
        return NULL;

    // set list lengths
    centers.len = center_length;
    peaks.len = peak_length;

    // printf("centers.len: %d\n", centers.len);
    // printf("peaks.len: %d\n", peaks.len);
    // printf("markerRadius: %f\n", markerRadius);
    // printf("distanceTolerance: %f\n", distanceTolerance);

    // allocate memories
    _allocateMemArray(&centers);
    _allocateMemArray(&peaks);

    // insert list values to memories
    for (i = 0; i < center_length; i++) {
        tuple_item = PyList_GetItem(center_list, i);
        if (!PyTuple_Check(tuple_item)){
            printf("bad!\n");
            (centers.co + i)->x = 0.0;
            (centers.co + i)->y = 0.0;    
        }
        temp_x = PyTuple_GetItem(tuple_item, 0);
        temp_y = PyTuple_GetItem(tuple_item, 1);
        // printf("coord x: %f, y: %f\n", PyFloat_AsDouble(temp_x), PyFloat_AsDouble(temp_y));
        (centers.co + i)->x = PyFloat_AsDouble(temp_x);
        (centers.co + i)->y = PyFloat_AsDouble(temp_y);
    }
    for (i = 0; i < peak_length; i++) {
        tuple_item = PyList_GetItem(peak_list, i);
        if (!PyTuple_Check(tuple_item)){
            printf("bad!\n");
            (peaks.co + i)->x = 0.0;
            (peaks.co + i)->y = 0.0;    
        }
        temp_x = PyTuple_GetItem(tuple_item, 0);
        temp_y = PyTuple_GetItem(tuple_item, 1);
        // printf("coord x: %f, y: %f\n", PyFloat_AsDouble(temp_x), PyFloat_AsDouble(temp_y));
        (peaks.co + i)->x = PyFloat_AsDouble(temp_x);
        (peaks.co + i)->y = PyFloat_AsDouble(temp_y);
    }

    _findCenter(
        centers,
        peaks, 
        &mcenters,
        markerRadius,
        distanceTolerance
        );

    _clusterCenters(mcenters, &clustered_centers, markerRadius);
    
    _getAccurateCenters(clustered_centers, &mcenters_filtered);

    // pack output array as list of tuples
    // printf("mcenters_filtered.len: %d\n", mcenters_filtered.len);
    PyObject* return_list = PyList_New(0);
    PyObject* return_item;
    for (i = 0; i < mcenters_filtered.len; i++) {
        return_item = Py_BuildValue("dd", (mcenters_filtered.co + i)->x, (mcenters_filtered.co + i)->y);
        // printf("return x: %f, y: %f\n", (mcenters_filtered.co + i)->x, (mcenters_filtered.co + i)->y);
        // printf("%d\n", PyList_SetItem(return_list, i, return_item));
        PyList_Append(return_list, return_item);
    }
    // return Py_BuildValue("i", 0);
    return return_list;
}

// forcestamp definition(names in python)
static PyMethodDef forcestampMethods[] = {
    { "findMarkerCenters", (PyCFunction)findMarkerCenters, METH_VARARGS, "Finds marker centers from circle centers and img peaks." },
    { NULL, NULL, 0, NULL }
};

// forcestamp definition struct
static struct PyModuleDef forcestamp_c = {
    PyModuleDef_HEAD_INIT,
    "forcestamp_c",
    "C implementation of forcestamp functions",
    -1,
    forcestampMethods
};

// Initializes forcestamp
PyMODINIT_FUNC PyInit_forcestamp_c(void)
{
    return PyModule_Create(&forcestamp_c);
}
