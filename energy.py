import sys
import math
import numpy as np
import ROOT as root
import numpy as np
from ROOT import gROOT, gSystem, TH2F, TTree, TFile, AddressOf, TLine, TMultiGraph, TEllipse, TH1F
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
import pandas as pd


class Energy:

    # Main Constructor
    def __init__(self, points, endpoints, calib, max_per):
        self.points = points
        self.endpoints = endpoints
        self.calib_table = calib
        self.max_per = max_per

    # Function to calculate the area cut out by the lines on the bin edge
    def calculate_weights_polygon(self, polygon1_arr, line1_arr, line2_arr, point_arr, verbose):
        polygon1 = Polygon(
            [(polygon1_arr[0, 0], polygon1_arr[1, 0]), (polygon1_arr[0, 1], polygon1_arr[1, 1]),
             (polygon1_arr[0, 2], polygon1_arr[1, 2]), (polygon1_arr[0, 3], polygon1_arr[1, 3]), ])
        line1 = LineString([(line1_arr[0, 0], line1_arr[1, 0]), (line1_arr[0, 1], line1_arr[1, 1])])
        line2 = LineString([(line2_arr[0, 0], line2_arr[1, 0]), (line2_arr[0, 1], line2_arr[1, 1])])
        point = Point(point_arr[0], point_arr[1])
        area_main_polygon = polygon1.area
        if verbose:
            x, y = polygon1.exterior.xy
        total_area = []
        areas = [0, 0, 0]
        cut_poly1 = split(polygon1, line1)
        for i in cut_poly1:
            if i.contains(point):
                cut_poly2 = split(i, line2)
                for j in cut_poly2:
                    if j.contains(point):
                        total_area.append(j.area)
                        areas[1] = j.area
                        if verbose:
                            cox, coy = j.exterior.xy
                    else:
                        total_area.append(j.area)
                        areas[2] = j.area
            else:
                total_area.append(i.area)
                areas[0] = i.area
        tot = np.array(total_area)
        if verbose:
            print('NOT NEEDED')
        return areas

    def energy_weighted(self):
        calibration_table = self.calib_table
        calibration_table.columns = ["chno", "xx", "yy", "par0", "par1", "chi"]
        data_array = self.points
        data_array_X = data_array[:, 0]
        data_array_Y = data_array[:, 1]
        data_array_Z = data_array[:, 2]
        data_array_Q = data_array[:, 3]
        line_vector_start_3d = [self.endpoints[0, 0], self.endpoints[0, 1], self.endpoints[0, 2]]
        line_vector_end_3d = [self.endpoints[1, 0], self.endpoints[1, 1], self.endpoints[1, 2]]
        line_vector_3d = np.subtract(line_vector_end_3d, line_vector_start_3d)
        unit_vector_3d = line_vector_3d / np.linalg.norm(line_vector_3d)
        line_length_3d = np.linalg.norm(line_vector_3d)
        line_vector_start_2d = [self.endpoints[0, 0], self.endpoints[0, 1]]
        line_vector_end_2d = [self.endpoints[1, 0], self.endpoints[1, 1]]
        line_vector_2d = np.subtract(line_vector_end_2d, line_vector_start_2d)
        line_length_2d = np.linalg.norm(line_vector_2d)
        unit_vector_2d = line_vector_2d / np.linalg.norm(line_vector_2d)
        num_bins = int(line_length_2d / settings.bin_size_un)
        # Finding the bin edges
        arr_bin_edges = np.zeros([num_bins + 2, 7])
        for i in range(0, num_bins + 2):
            arr_bin_edges[i, 0] = i
            next_vector = line_vector_start_2d + i * settings.bin_size_un * unit_vector_2d
            arr_bin_edges[i, 1] = next_vector[0]
            arr_bin_edges[i, 2] = next_vector[1]
            if i > 0:
                x3, y3, x4, y4 = self.perp_line(line_vector_start_2d[0], next_vector[0], line_vector_start_2d[1], next_vector[1], settings.length_perp_bin)
                arr_bin_edges[i, 3] = x3
                arr_bin_edges[i, 4] = y3
                arr_bin_edges[i, 5] = x4
                arr_bin_edges[i, 6] = y4
            else:
                x3, y3, x4, y4 = self.perp_line(line_vector_end_2d[0], line_vector_start_2d[0], line_vector_end_2d[1], line_vector_start_2d[1], settings.length_perp_bin)
                arr_bin_edges[i, 3] = x3
                arr_bin_edges[i, 4] = y3
                arr_bin_edges[i, 5] = x4
                arr_bin_edges[i, 6] = y4
        histogram_array_list = []
        for j in range(0, len(data_array_X)):
            p = Point(data_array_X[j], data_array_Y[j])
            ab = LineString([(line_vector_start_2d[0], line_vector_start_2d[1]), (line_vector_end_2d[0], line_vector_end_2d[1])])
            dist = ab.project(p)
            (prx, pry) = ab.interpolate(dist).coords.xy
            projection_next = np.subtract([prx[0], pry[0]], line_vector_start_2d)
            mag_projection_next = np.linalg.norm(projection_next)
            possible_id = int(mag_projection_next / settings.bin_size_un)
            polygon1_arr = np.array(
                [[data_array_X[j] - 1, data_array_X[j] + 1, data_array_X[j] + 1, data_array_X[j] - 1],
                 [data_array_Y[j] - 1, data_array_Y[j] - 1, data_array_Y[j] + 1, data_array_Y[j] + 1]])
            line1_arr = np.array([[arr_bin_edges[possible_id, 3], arr_bin_edges[possible_id, 5]],
                                  [arr_bin_edges[possible_id, 4], arr_bin_edges[possible_id, 6]]])
            line2_arr = np.array([[arr_bin_edges[possible_id + 1, 3], arr_bin_edges[possible_id + 1, 5]],
                                  [arr_bin_edges[possible_id + 1, 4], arr_bin_edges[possible_id + 1, 6]]])
            point_arr = [data_array_X[j], data_array_Y[j]]
            areas = self.calculate_weights_polygon(polygon1_arr, line1_arr, line2_arr, point_arr, False)
            #Calibration
            pad_info_single = calibration_table.loc[(calibration_table['xx'] == int(data_array_X[j]/settings.x_conversion_factor)) & (calibration_table['yy'] == int(data_array_Y[j]/settings.y_conversion_factor))]
            try:
                if settings.simulation_events:
                    Qvox_value = data_array_Q[j]
                    # print('Energy Simulated')
                else:
                    if pad_info_single['chi'].values[0] < 8000:
                        Qvox_value = (data_array_Q[j] - pad_info_single['par0'].values[0]) / pad_info_single['par1'].values[0]
                    else:
                        Qvox_value = 0
                Qvox = round(float(Qvox_value), 2)
                #Qvox = data_array_Q[j]
            except:
                Qvox = data_array_Q[j]
            charge_info = Qvox
            histogram_array_list.append([possible_id - 1, (areas[0] / settings.area_total_pad) * charge_info])
            histogram_array_list.append([possible_id, (areas[1] / settings.area_total_pad) * charge_info])
            histogram_array_list.append([possible_id + 1, (areas[2] / settings.area_total_pad) * charge_info])
        histogram_array = np.array(histogram_array_list)
        histogram_array_new = np.zeros([len(np.unique(arr_bin_edges[:, 0])), 6])
        counter = 0
        ratio_3d = round(line_length_3d / line_length_2d, 2)
        for r in np.unique(arr_bin_edges[:, 0]):
            histogram_array_new_id = histogram_array[:, 0]
            result = np.where(histogram_array_new_id == r)
            result_charge = histogram_array[result, 1]
            ve = np.where(arr_bin_edges[:, 0] == r)
            histogram_array_new[counter, 0] = r + (settings.bin_size_un / 2)
            vector_id_2d = line_vector_start_2d + ((r * settings.bin_size_un) + (settings.bin_size_un / 2)) * unit_vector_2d
            vector_id_3d = line_vector_start_3d + ((r * settings.bin_size_un) + (settings.bin_size_un / 2)) * ratio_3d * unit_vector_3d
            point_bin_2d = np.linalg.norm(np.array(vector_id_2d)-np.array(line_vector_start_2d))
            point_bin_3d = ratio_3d*point_bin_2d
            histogram_array_new[counter, 1] = arr_bin_edges[ve, 1]
            histogram_array_new[counter, 2] = arr_bin_edges[ve, 2]
            histogram_array_new[counter, 3] = arr_bin_edges[ve, 2]
            histogram_array_new[counter, 4] = np.sum(result_charge)
            histogram_array_new[counter, 5] = round(point_bin_3d, 2)
            counter += 1
        new_energy = histogram_array_new[:, 4]
        new_position = histogram_array_new[:, 5]
        new_energy = np.append(new_energy, 0)
        new_position = np.append(new_position, new_position[-1] + (new_position[1] - new_position[0]))
        if line_length_3d > settings.line_length_threshold:
            golay_window = settings.savitzky_golay_window_large
        else:
            golay_window = settings.savitzky_golay_window_small
        try:
            if len(new_energy) > golay_window:
                fit_energy_ = savgol_filter(new_energy, golay_window, 3, mode='interp')
            else:
                if len(new_energy) % 2 == 0:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 1, 1,
                                                mode='interp')
                else:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 2, 1,
                                                mode='interp')
        except:
            fit_energy_ = new_energy
        try:
            cs_position = np.linspace(new_position.min(), new_position.max(), 5000)
            cs_function = CubicSpline(new_position, fit_energy_)
            cs_energy = cs_function(cs_position)
            threshold_peak_height = settings.threshold_peaks * np.average(cs_energy)
            peaks, _ = find_peaks(cs_energy, height=threshold_peak_height)
            if len(peaks) != 0:
                peak_lines = cs_energy[peaks]
            else:
                peak_lines = cs_energy[-1]
            range_max = np.max(peak_lines)
            idx_ = Energy.find_nearest(cs_energy, range_max)
            max_energy = cs_energy[idx_]
            max_position = cs_position[idx_]
            fit_energy_new = cs_energy[idx_:]
            positions_ = cs_position
            positions_new = positions_[idx_:]
            arr_min_ = np.abs(fit_energy_new - (self.max_per * max_energy))
            idx_low = -1
            idx_low_new = np.argmin(arr_min_)
            idx_low = idx_low_new

             #Introduced to find the intersection points
            f = cs_energy
            g = self.max_per * max_energy*np.ones(cs_position.shape)
            idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
            idx_max = max(idx)

            #### PART REPLACED WHICH FINDS THE VALUE FROM BACKWARDS
            # print('FOUND ENERGY->', fit_energy_new(idx_low_new), settings.max_percentage * max_energy)
            # print('Printing arr_min_values', arr_min_flip[idx_low])
            # for ind, after_val in enumerate(arr_min_flip):
            #     if after_val < settings.end_energy_diff:
            #         idx_low = len(arr_min_) - 1 - ind
            #         print("Inside the Loop")
            #         break
            # print('Printing arr_min_values', arr_min_flip[idx_low])
            vector_id_3d_final = line_vector_start_3d + positions_new[idx_low] * unit_vector_3d
            delta_z = vector_id_3d_final[2] - line_vector_start_3d[2]
            r2d = np.linalg.norm(np.array(vector_id_3d_final[0:2]) - np.array(line_vector_start_3d[0:2]))
        except:
            return line_length_2d, line_vector_end_3d[2]-line_vector_start_3d[2], np.mean(histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), np.mean(
                histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), histogram_array_new[:,
                                                                                5], histogram_array_new[:,
                                                                                    4], histogram_array_new[:,
                                                                                        5], histogram_array_new[:, 4]
        else:
            # R2D, DeltaZ, Pos_end, Energy_end, Pos_max, Energy_max, Pos_bin, Energy_bin, Pos_bin_smoothed, Energy_bin_smoothed
            # return r2d, delta_z, positions_new[idx_low], fit_energy_new[idx_low], positions_[idx_], max_energy, histogram_array_new[:,
            #                                                                                       5], histogram_array_new[
            #                                                                                           :,
            #                                                                                           4], cs_position, cs_energy
            return r2d, delta_z, cs_position[idx_max], cs_energy[idx_max], positions_[idx_], max_energy, histogram_array_new[:,
                                                                                                  5], histogram_array_new[
                                                                                                      :,
                                                                                                      4], cs_position, cs_energy


    def perp_line(self, x1, x2, y1, y2, cd_length):
        ab = LineString([(x1, y1), (x2, y2)])
        left = ab.parallel_offset(cd_length / 2, 'left')
        right = ab.parallel_offset(cd_length / 2, 'right')
        c = left.boundary[1]
        d = right.boundary[0]
        cd = LineString([c, d])
        return c.x, c.y, d.x, d.y

    # Instance Method
    def energy_unweighted(self):
        calibration_table = self.calib_table
        calibration_table.columns = ["chno", "xx", "yy", "par0", "par1", "chi"]
        line_length = Energy.vector_length(self.endpoints)
        num_bins = int(line_length / settings.bin_size_un)
        line_vector_start = [self.endpoints[0, 0], self.endpoints[0, 1], self.endpoints[0, 2]]
        line_vector_end = [self.endpoints[1, 0], self.endpoints[1, 1], self.endpoints[1, 2]]
        line_vector = np.subtract(line_vector_end, line_vector_start)
        unit_vector_ = line_vector / line_length
        arr_bin_edges = np.zeros([num_bins + 2, 4])
        # Finding the bin edges
        for i in range(0, num_bins + 2):
            arr_bin_edges[i, 0] = i
            next_vector = line_vector_start + i * settings.bin_size_un * unit_vector_
            arr_bin_edges[i, 1] = next_vector[0]
            arr_bin_edges[i, 2] = next_vector[1]
            arr_bin_edges[i, 3] = next_vector[2]
        # Projecting the points on to the Line.
        histogram_array = np.zeros([len(self.points[:, 0]), 2])
        for j in range(0, len(self.points[:, 0])):
            vector_point = np.array(
                [self.points[j, 0] - self.endpoints[0, 0], self.points[j, 1] - self.endpoints[0, 1],
                 self.points[j, 2] - self.endpoints[0, 2]])
            projection_vector = line_vector_start + np.dot(vector_point, unit_vector_) * unit_vector_
            projection_next = np.subtract(projection_vector, line_vector_start)
            mag_projection_next = np.linalg.norm(projection_next)
            possible_id = int(mag_projection_next / settings.bin_size_un)
            histogram_array[j, 0] = possible_id
            # Calibration
            pad_info_single = calibration_table.loc[
                (calibration_table['xx'] == int(self.points[j, 0] / settings.x_conversion_factor)) & (
                            calibration_table['yy'] == int(self.points[j, 1] / settings.y_conversion_factor))]
            try:
                if settings.simulation_events:
                    Qvox_value = self.points[j, 3]
                else:
                    if pad_info_single['chi'].values[0] < 8000:
                        Qvox_value = (self.points[j, 3] - pad_info_single['par0'].values[0]) / pad_info_single['par1'].values[0]
                    else:
                        Qvox_value = 0
                Qvox = round(float(Qvox_value), 2)
            except:
                Qvox = self.points[j, 3]
            charge_info = Qvox
            histogram_array[j, 1] = charge_info
        histogram_array_new = np.zeros([len(np.unique(arr_bin_edges[:, 0])), 6])
        counter = 0
        for r in np.unique(arr_bin_edges[:, 0]):
            histogram_array_new_id = histogram_array[:, 0]
            result = np.where(histogram_array_new_id == r)
            result_charge = histogram_array[result, 1]
            ve = np.where(arr_bin_edges[:, 0] == r)
            histogram_array_new[counter, 0] = r + (settings.bin_size_un / 2)
            vector_id = line_vector_start + ((r * settings.bin_size_un) + (settings.bin_size_un / 2)) * unit_vector_
            point_bin_ = math.sqrt(
                ((vector_id[0] - line_vector_start[0]) ** 2) + ((vector_id[1] - line_vector_start[1]) ** 2) + (
                        (vector_id[2] - line_vector_start[2]) ** 2))
            histogram_array_new[counter, 1] = arr_bin_edges[ve, 1]
            histogram_array_new[counter, 2] = arr_bin_edges[ve, 2]
            histogram_array_new[counter, 3] = arr_bin_edges[ve, 3]
            histogram_array_new[counter, 4] = np.sum(result_charge)
            histogram_array_new[counter, 5] = round(point_bin_, 2)
            counter += 1
        new_energy = histogram_array_new[:, 4]
        new_position = histogram_array_new[:, 5]
        new_energy = np.append(new_energy, 0)
        new_position = np.append(new_position, new_position[-1] + (new_position[1] - new_position[0]))
        if line_length > settings.line_length_threshold:
            golay_window = settings.savitzky_golay_window_large
        else:
            golay_window = settings.savitzky_golay_window_small
        try:
            if len(new_energy) > golay_window:
                fit_energy_ = savgol_filter(new_energy, golay_window, 3, mode='interp')
            else:
                if len(new_energy) % 2 == 0:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 1, 1,
                                                mode='interp')
                else:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 2, 1,
                                                mode='interp')
        except:
            fit_energy_ = new_energy
        try:
            cs_position = np.linspace(new_position.min(), new_position.max(), 1000)
            cs_function = CubicSpline(new_position, fit_energy_)
            cs_energy = cs_function(cs_position)
            threshold_peak_height = settings.threshold_peaks * np.average(cs_energy)
            peaks, _ = find_peaks(cs_energy, height=threshold_peak_height)
            if len(peaks) != 0:
                peak_lines = cs_energy[peaks]
            else:
                peak_lines = cs_energy[-1]
            range_max = np.max(peak_lines)
            idx_ = Energy.find_nearest(cs_energy, range_max)
            max_energy = cs_energy[idx_]
            fit_energy_new = cs_energy[idx_:]
            positions_ = cs_position
            positions_new = positions_[idx_:]
            arr_min_ = np.abs(fit_energy_new - (self.max_per * max_energy))
            arr_min_flip = np.flip(arr_min_)
            idx_low = -1
            idx_low_new = np.argmin(arr_min_)
            idx_low = idx_low_new


             #Introduced to find the intersection points
            f = cs_energy
            g = self.max_per * max_energy*np.ones(cs_position.shape)
            idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
            idx_max = max(idx)

            #### PART REPLACED WHICH FINDS THE VALUE FROM BACKWARDS
            #print('FOUND ENERGY->', fit_energy_new(idx_low_new), settings.max_percentage * max_energy)
            # print('Printing arr_min_values', arr_min_flip[idx_low])
            # for ind, after_val in enumerate(arr_min_flip):
            #     if after_val < settings.end_energy_diff:
            #         idx_low = len(arr_min_) - 1 - ind
            #         print("Inside the Loop")
            #         break
            # print('Printing arr_min_values', arr_min_flip[idx_low])
            vector_id_3d_final = line_vector_start + positions_new[idx_low] * unit_vector_
            delta_z = vector_id_3d_final[2] - line_vector_start[2]
            r2d = np.linalg.norm(np.array(vector_id_3d_final[0:2]) - np.array(line_vector_start[0:2]))
        except:
            print('ENCOUNTERED EXCEPTION')
            return line_length, line_vector_end[2]-line_vector_start[2], np.mean(histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), np.mean(
                histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), histogram_array_new[:,
                                                                                5], histogram_array_new[:,
                                                                                    4], histogram_array_new[:,
                                                                                        5], histogram_array_new[:,
                                                                                            4]
        else:
            # R2D, Delta, Pos_end, Energy_end, Pos_max, Energy_max, Pos_bin, Energy_bin, Pos_bin_smoothed, Energy_bin_smoothed
            # return r2d, delta_z, positions_new[idx_low], fit_energy_new[idx_low], positions_[
                # idx_], max_energy, histogram_array_new[:,
                #                    5], histogram_array_new[
                #                        :,
                #                    4], cs_position, cs_energy
            return r2d, delta_z, cs_position[idx_max], cs_energy[idx_max], positions_[idx_], max_energy, histogram_array_new[:,
                                                                                                  5], histogram_array_new[
                                                                                                      :,
                                                                                                      4], cs_position, cs_energy


    def energy_weighted_no_calib(self):
        calibration_table = self.calib_table
        calibration_table.columns = ["chno", "xx", "yy", "par0", "par1", "chi"]
        data_array = self.points
        data_array_X = data_array[:, 0]
        data_array_Y = data_array[:, 1]
        data_array_Z = data_array[:, 2]
        data_array_Q = data_array[:, 3]
        line_vector_start_3d = [self.endpoints[0, 0], self.endpoints[0, 1], self.endpoints[0, 2]]
        line_vector_end_3d = [self.endpoints[1, 0], self.endpoints[1, 1], self.endpoints[1, 2]]
        line_vector_3d = np.subtract(line_vector_end_3d, line_vector_start_3d)
        unit_vector_3d = line_vector_3d / np.linalg.norm(line_vector_3d)
        line_length_3d = np.linalg.norm(line_vector_3d)
        line_vector_start_2d = [self.endpoints[0, 0], self.endpoints[0, 1]]
        line_vector_end_2d = [self.endpoints[1, 0], self.endpoints[1, 1]]
        line_vector_2d = np.subtract(line_vector_end_2d, line_vector_start_2d)
        line_length_2d = np.linalg.norm(line_vector_2d)
        unit_vector_2d = line_vector_2d / np.linalg.norm(line_vector_2d)
        num_bins = int(line_length_2d / settings.bin_size_un)
        # Finding the bin edges
        arr_bin_edges = np.zeros([num_bins + 2, 7])
        for i in range(0, num_bins + 2):
            arr_bin_edges[i, 0] = i
            next_vector = line_vector_start_2d + i * settings.bin_size_un * unit_vector_2d
            arr_bin_edges[i, 1] = next_vector[0]
            arr_bin_edges[i, 2] = next_vector[1]
            if i > 0:
                x3, y3, x4, y4 = self.perp_line(line_vector_start_2d[0], next_vector[0], line_vector_start_2d[1], next_vector[1], settings.length_perp_bin)
                arr_bin_edges[i, 3] = x3
                arr_bin_edges[i, 4] = y3
                arr_bin_edges[i, 5] = x4
                arr_bin_edges[i, 6] = y4
            else:
                x3, y3, x4, y4 = self.perp_line(line_vector_end_2d[0], line_vector_start_2d[0], line_vector_end_2d[1], line_vector_start_2d[1], settings.length_perp_bin)
                arr_bin_edges[i, 3] = x3
                arr_bin_edges[i, 4] = y3
                arr_bin_edges[i, 5] = x4
                arr_bin_edges[i, 6] = y4
        histogram_array_list = []
        for j in range(0, len(data_array_X)):
            p = Point(data_array_X[j], data_array_Y[j])
            ab = LineString([(line_vector_start_2d[0], line_vector_start_2d[1]), (line_vector_end_2d[0], line_vector_end_2d[1])])
            dist = ab.project(p)
            (prx, pry) = ab.interpolate(dist).coords.xy
            projection_next = np.subtract([prx[0], pry[0]], line_vector_start_2d)
            mag_projection_next = np.linalg.norm(projection_next)
            possible_id = int(mag_projection_next / settings.bin_size_un)
            polygon1_arr = np.array(
                [[data_array_X[j] - 1, data_array_X[j] + 1, data_array_X[j] + 1, data_array_X[j] - 1],
                 [data_array_Y[j] - 1, data_array_Y[j] - 1, data_array_Y[j] + 1, data_array_Y[j] + 1]])
            line1_arr = np.array([[arr_bin_edges[possible_id, 3], arr_bin_edges[possible_id, 5]],
                                  [arr_bin_edges[possible_id, 4], arr_bin_edges[possible_id, 6]]])
            line2_arr = np.array([[arr_bin_edges[possible_id + 1, 3], arr_bin_edges[possible_id + 1, 5]],
                                  [arr_bin_edges[possible_id + 1, 4], arr_bin_edges[possible_id + 1, 6]]])
            point_arr = [data_array_X[j], data_array_Y[j]]
            areas = self.calculate_weights_polygon(polygon1_arr, line1_arr, line2_arr, point_arr, False)
            #Calibration
            pad_info_single = calibration_table.loc[(calibration_table['xx'] == int(data_array_X[j]/settings.x_conversion_factor)) & (calibration_table['yy'] == int(data_array_Y[j]/settings.y_conversion_factor))]
            try:
                if settings.simulation_events:
                    Qvox_value = data_array_Q[j]
                else:
                    if pad_info_single['chi'].values[0] < 8000:
                        Qvox_value = (data_array_Q[j] - pad_info_single['par0'].values[0]) / pad_info_single['par1'].values[0]
                    else:
                        Qvox_value = 0
                Qvox = round(float(Qvox_value), 2)
                #Qvox = data_array_Q[j]
            except:
                Qvox = data_array_Q[j]
            charge_info = Qvox
            histogram_array_list.append([possible_id - 1, (areas[0] / settings.area_total_pad) * charge_info])
            histogram_array_list.append([possible_id, (areas[1] / settings.area_total_pad) * charge_info])
            histogram_array_list.append([possible_id + 1, (areas[2] / settings.area_total_pad) * charge_info])
        histogram_array = np.array(histogram_array_list)
        histogram_array_new = np.zeros([len(np.unique(arr_bin_edges[:, 0])), 6])
        counter = 0
        ratio_3d = round(line_length_3d / line_length_2d, 2)
        for r in np.unique(arr_bin_edges[:, 0]):
            histogram_array_new_id = histogram_array[:, 0]
            result = np.where(histogram_array_new_id == r)
            result_charge = histogram_array[result, 1]
            ve = np.where(arr_bin_edges[:, 0] == r)
            histogram_array_new[counter, 0] = r + (settings.bin_size_un / 2)
            vector_id_2d = line_vector_start_2d + ((r * settings.bin_size_un) + (settings.bin_size_un / 2)) * unit_vector_2d
            vector_id_3d = line_vector_start_3d + ((r * settings.bin_size_un) + (settings.bin_size_un / 2)) * ratio_3d * unit_vector_3d
            point_bin_2d = np.linalg.norm(np.array(vector_id_2d)-np.array(line_vector_start_2d))
            point_bin_3d = ratio_3d*point_bin_2d
            histogram_array_new[counter, 1] = arr_bin_edges[ve, 1]
            histogram_array_new[counter, 2] = arr_bin_edges[ve, 2]
            histogram_array_new[counter, 3] = arr_bin_edges[ve, 2]
            histogram_array_new[counter, 4] = np.sum(result_charge)
            histogram_array_new[counter, 5] = round(point_bin_3d, 2)
            counter += 1
        new_energy = histogram_array_new[:, 4]
        new_position = histogram_array_new[:, 5]
        new_energy = np.append(new_energy, 0)
        new_position = np.append(new_position, new_position[-1] + (new_position[1] - new_position[0]))
        if line_length_3d > settings.line_length_threshold:
            golay_window = settings.savitzky_golay_window_large
        else:
            golay_window = settings.savitzky_golay_window_small
        try:
            if len(new_energy) > golay_window:
                fit_energy_ = savgol_filter(new_energy, golay_window, 3, mode='interp')
            else:
                if len(new_energy) % 2 == 0:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 1, 1,
                                                mode='interp')
                else:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 2, 1,
                                                mode='interp')
        except:
            fit_energy_ = new_energy
        try:
            cs_position = np.linspace(new_position.min(), new_position.max(), 5000)
            cs_function = CubicSpline(new_position, fit_energy_)
            cs_energy = cs_function(cs_position)
            threshold_peak_height = settings.threshold_peaks * np.average(cs_energy)
            peaks, _ = find_peaks(cs_energy, height=threshold_peak_height)
            if len(peaks) != 0:
                peak_lines = cs_energy[peaks]
            else:
                peak_lines = cs_energy[-1]
            range_max = np.max(peak_lines)
            idx_ = Energy.find_nearest(cs_energy, range_max)
            max_energy = cs_energy[idx_]
            max_position = cs_position[idx_]
            fit_energy_new = cs_energy[idx_:]
            positions_ = cs_position
            positions_new = positions_[idx_:]
            arr_min_ = np.abs(fit_energy_new - (self.max_per * max_energy))
            idx_low = -1
            idx_low_new = np.argmin(arr_min_)
            idx_low = idx_low_new

             #Introduced to find the intersection points
            f = cs_energy
            g = self.max_per * max_energy*np.ones(cs_position.shape)
            idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
            idx_max = max(idx)

            #### PART REPLACED WHICH FINDS THE VALUE FROM BACKWARDS
            # print('FOUND ENERGY->', fit_energy_new(idx_low_new), settings.max_percentage * max_energy)
            # print('Printing arr_min_values', arr_min_flip[idx_low])
            # for ind, after_val in enumerate(arr_min_flip):
            #     if after_val < settings.end_energy_diff:
            #         idx_low = len(arr_min_) - 1 - ind
            #         print("Inside the Loop")
            #         break
            # print('Printing arr_min_values', arr_min_flip[idx_low])
            vector_id_3d_final = line_vector_start_3d + positions_new[idx_low] * unit_vector_3d
            delta_z = vector_id_3d_final[2] - line_vector_start_3d[2]
            r2d = np.linalg.norm(np.array(vector_id_3d_final[0:2]) - np.array(line_vector_start_3d[0:2]))
        except:
            return line_length_2d, line_vector_end_3d[2]-line_vector_start_3d[2], np.mean(histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), np.mean(
                histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), histogram_array_new[:,
                                                                                5], histogram_array_new[:,
                                                                                    4], histogram_array_new[:,
                                                                                        5], histogram_array_new[:, 4]
        else:
            # R2D, DeltaZ, Pos_end, Energy_end, Pos_max, Energy_max, Pos_bin, Energy_bin, Pos_bin_smoothed, Energy_bin_smoothed
            # return r2d, delta_z, positions_new[idx_low], fit_energy_new[idx_low], positions_[idx_], max_energy, histogram_array_new[:,
            #                                                                                       5], histogram_array_new[
            #                                                                                           :,
            #                                                                                           4], cs_position, cs_energy
            return r2d, delta_z, cs_position[idx_max], cs_energy[idx_max], positions_[idx_], max_energy, histogram_array_new[:,
                                                                                                  5], histogram_array_new[
                                                                                                      :,
                                                                                                      4], cs_position, cs_energy

    #Static Method
    def vector_length(endpts):
        return math.sqrt(((endpts[1, 0] - endpts[0, 0]) ** 2) + ((endpts[1, 1] - endpts[0, 1]) ** 2) + (
                (endpts[1, 2] - endpts[0, 2]) ** 2))

    # Static Method
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


