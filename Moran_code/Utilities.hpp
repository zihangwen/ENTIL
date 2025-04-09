//
//  Created by Zihang Wen on 3/5/23.
//  Copyright © 2023 Zihang Wen. All rights reserved.
//

#pragma once

#include <random>
#include <ctime>

#include <iostream>
#include <fstream>
#include <string>

#include <vector>
#include <stdexcept>
#include <set>
#include <cmath>

using namespace std;

mt19937 generator(time(0));

std::vector<double> accumulate_sum(std::vector<double> p) {
   std::vector<double> p_accu(p.size());
   p_accu[0] = p[0];
   for (int i = 1; i < p.size(); i++) {
       p_accu[i] = p_accu[i-1] + p[i];
   }
   return p_accu;
}

std::vector<double> accumulate_sum(std::vector<int> p) {
   std::vector<double> p_accu(p.size());
   p_accu[0] = p[0];
   for (int i = 1; i < p.size(); i++) {
       p_accu[i] = p_accu[i-1] + p[i];
   }
   return p_accu;
}

// int random_choice_single(std::vector<double> p_accu) {

//     int vector_size = p_accu.size();
//     int j = 0;

//     std::uniform_real_distribution<double> rand_rd(0.0, p_accu[vector_size - 1]);
//     double random_p = rand_rd(generator);
//     while (random_p > p_accu[j]) {
//         j++;
//     }
//     return j;
// }

int random_choice_single(std::vector<double> p_accu) {
   int j = 0;

   std::uniform_real_distribution<double> rand_rd(0.0, p_accu.back());
   double random_p = rand_rd(generator);
   while (random_p > p_accu[j]) {
       j++;
   }
   return j;
}

int random_choice_single(std::vector<int> p_accu) {
   int j = 0;

   std::uniform_real_distribution<double> rand_rd(0.0, p_accu.back());
   double random_p = rand_rd(generator);
   while (random_p > p_accu[j]) {
       j++;
   }
   return j;
}

std::vector<int> random_choice(std::vector<int> origin_vector, std::vector<double> p, int sample_size, bool replace) {
   int sampled_index;
   std::vector<int> sampled_vec;
   std::vector<double> p_accu;

   if (replace == true) {
       p_accu = accumulate_sum(p);
       for (int i = 0; i < sample_size; i++) {
           sampled_vec.push_back(origin_vector[random_choice_single(p_accu)]);
       }
   } else {
       for (int i = 0; i < sample_size; i++) {
           p_accu = accumulate_sum(p);
           sampled_index = random_choice_single(p_accu);
           sampled_vec.push_back(origin_vector[sampled_index]);
           origin_vector.erase(origin_vector.begin() + sampled_index);
           p.erase(p.begin() + sampled_index);
       }
   }
   return sampled_vec;
}

std::vector<int> random_choice(int num_ov, std::vector<double> p, int sample_size, bool replace) {
   std::vector<int> origin_vector;
   for (int i = 0; i < num_ov; ++i) {
       // fitness_bar[i] /= sum_fitness;
       origin_vector.push_back(i);
   }

   return random_choice(origin_vector, p, sample_size, replace);
}

double poisson_sample(double mu) {
   std::poisson_distribution<int> rand_poi(mu);
   int random_poi = rand_poi(generator);
   return random_poi;
}

int binomial_sample(int n, double p) {
   std::binomial_distribution<int> rand_bi(n, p);
   int random_bi = rand_bi(generator);
   return random_bi;
}

int uniform_int_sample(int a, int b) {
   std::uniform_int_distribution<int> rand_ui(a, b);
   int random_ui = rand_ui(generator);
   return random_ui;
}

double uniform_sample(double a, double b) {
   std::uniform_real_distribution<double> rand_uni(a, b);
   double random_uni =rand_uni(generator);
   return random_uni;
}