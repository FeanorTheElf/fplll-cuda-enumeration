import math
import numpy as np

mt = [
(9831350, 40, [90, 60, 175, 30, -106, 47, 57, -72, 35, -220, -169, 77, -29, -7, -57, 117, -41, 55, -31, -69, -6, 118, 27, 31, 17, 145, -162, 36, 94, 243, -297, -81, -69, -174, 63, 111, -230, 0, 90, -25, -8]),
(7664355, 30, [-93, -96, 60, 25, 110, -37, -12, -109, -176, -83, 107, 98, -8, -229, -23, -24, -112, -28, -149, 39, 41, -109, 11, 114, 166, -214, -9, -49, -229, 214, 32, -13, 57, 33, 200, -37, 182, 141, -122, 105, 154]),
(1414452, 6, [-36, 57, -124, 6, 78, 24, -40, -20, 45, 128, -111, -152, 167, -138, -7, 97, 78, 81, -14, 16, -116, -40, -198, -64, 114, -135, -77, -98, 18, -49, 171, 76, -184, 154, -162, 57, 44, 293, -67, -4, -100]),
(19302773, 79, [-79, -96, -31, -101, 19, -30, 16, -77, 43, -79, 117, -77, 55, -48, 63, -86, 35, -17, -69, 74, 165, 43, 84, 34, -189, 1, -114, 76, 21, 238, -110, 66, 81, 81, 19, -124, 218, -116, -126, -83, -147, 79]),
(8266276, 44, [34, 51, 36, -24, 41, 128, -94, -151, 85, 104, -58, 216, 5, 54, -52, -173, -139, 61, -152, -9, -16, 26, 59, 166, 36, -61, -73, 23, 77, 6, 20, -175, 56, -65, -28, 165, 143, -7, -3, -101, -18, 216]),
(6982609, 29, [-13, 149, -12, 30, -13, -39, 175, -29, 16, -76, 49, 184, -100, 99, -64, -95, -28, -113, -224, 100, -50, 50, -79, -193, -47, 40, 148, 3, 118, -43, 109, 41, 36, -135, 80, -24, -93, -105, -130, -135, 24, 34]),
(56466745, 230, [-75, -12, -36, -43, -182, 14, -82, -34, 125, -103, -129, 99, -21, -134, 49, -168, -33, -5, -8, -5, -159, 85, -53, -28, -192, 87, 75, 38, 94, 142, 30, -50, 275, 22, 138, 86, -165, -5, -16, 31, 178, -137]),
(30604313, 117, [47, -34, 33, 40, -145, 46, 43, -110, -20, -173, 159, -83, 2, 22, -67, 90, 27, -4, 156, -68, -62, 88, -13, -131, -74, -60, 26, 66, -10, 71, 185, -45, -75, -149, 13, -85, 8, 91, 124, 62, 56, 131, -73]),
(8298137, 36, [-41, 78, -80, -19, 61, -5, 0, -25, 36, -124, -100, -81, 133, -62, 113, -2, -198, 31, 13, 52, -67, -114, -16, -10, -34, 12, 72, 118, 10, 86, 87, -55, 55, 53, -77, 54, 176, 20, 37, -28, -72, -194, -75]),
(54029363, 207, [-213, 48, -28, -76, -40, 14, -67, -19, 178, 80, 55, 178, -70, 63, -68, -40, 27, -119, -56, 33, 52, 8, 270, 47, -145, -42, 40, -71, -27, 16, 112, -18, -7, -68, -98, 44, 17, 28, -13, 160, -15, -1, 54]),
(26042112, 107, [-64, 72, -146, 85, 2, -96, -161, 65, -49, 65, 54, 45, -50, -3, -53, -64, 175, 22, 27, 90, 31, 107, 57, -132, -11, -130, -58, -1, -93, 42, -105, -145, -8, -90, 49, 45, 69, 18, -122, 72, 42, 66, 261]),
(49577550, 203, [9, -24, 49, -21, 50, -110, 4, 46, 28, 72, -43, 67, -135, -60, 137, 146, 11, -116, -82, 1, 1, -56, 9, 27, 14, 8, 3, -178, -77, -36, 45, 110, 122, -17, 22, -80, -57, 21, 57, 6, 25, 37, 25, -188]),
(155197520, 750, [-70, 19, -60, 10, -70, 106, 116, 64, -42, -61, -73, -27, 43, 101, 123, 81, -65, -43, 58, 4, -26, -49, 83, -89, -75, -6, 47, -176, 73, 8, -15, 72, -52, 53, 20, -15, 24, 132, 1, -4, 68, -208, 20, -32]),
(105285734, 404, [-2, -112, -35, -73, -92, -63, -23, 150, 0, -18, -58, 120, 28, 30, -93, -72, 119, -56, -103, -14, 28, 27, 73, -65, 39, 83, 55, -77, 119, -37, -27, -88, -102, -141, 41, 52, -51, -93, -51, -64, 85, 126, -17, 76]),
(220560807, 927, [-21, 32, 0, -44, 114, 23, -62, 51, -128, 67, -151, -155, 9, -97, 18, -31, 37, 10, 21, -128, -75, 108, 54, 63, 21, -50, -72, 28, 97, -77, -31, 145, 17, -106, -35, 111, 39, 83, -37, 59, -89, 46, -7, -29]),
(120665488, 785, [-62, 61, -43, 29, -54, -117, 19, 36, 31, -38, 8, 75, 7, 101, 71, 7, 17, -6, 62, -142, 68, 148, 36, -43, -41, -48, 17, -7, 12, -1, -106, 6, 70, -17, -27, -18, 75, -62, 59, -160, 31, -63, 0, 44, 118]),
(409721730, 2070, [13, -12, -104, -10, 34, 22, -64, 6, 18, 56, 40, -15, -6, -49, 18, -95, -57, 113, -24, -13, 121, 24, -85, -103, -93, -142, 2, -66, -61, 24, 90, -45, -56, 38, 31, -12, 93, -9, -2, -40, 62, 120, -74, 101, 106]),
(161443995, 787, [-3, -67, 44, 17, -44, 10, 68, 41, 24, 10, -10, 17, -64, -91, 11, 51, -1, -18, 11, -116, -109, 34, -79, -26, -32, -8, 64, -12, 47, 33, -1, -90, 126, 1, 31, 145, -114, 132, 34, 19, -134, -67, 70, -17, 152]),
(88576007, 349, [41, 11, -17, 51, 104, 40, 125, 86, 1, -136, -47, -46, 16, 4, -13, 86, 62, 22, 53, -54, -96, 11, -50, 111, 106, 20, 27, 23, 6, -79, -15, 36, 16, -13, 10, -56, -115, 93, -106, -59, 0, -38, -95, -99, -34]),
(276733304, 1086, [25, -1, -37, 41, 47, -63, 101, -5, 17, -35, 40, -18, 12, 46, -89, 50, 67, -17, -95, -78, -23, 114, -88, 52, -99, 4, 49, -75, 107, 21, -11, -82, 105, 86, 19, 19, -32, -64, -45, -4, 11, -103, 14, -14, -22, -22]),
(679273160, 3393, [4, 5, -71, 25, 29, -36, -13, -44, -44, -86, 44, -44, -31, 42, -60, 78, 93, -96, 56, -18, -103, 24, -5, 53, -74, 131, 19, 5, -16, 17, -61, 57, -43, -126, -16, -18, -11, 1, 13, -29, -122, 72, -1, 59, 22, -30]),
(741157595, 2883, [-35, 146, 95, 94, -21, -25, 54, -22, -14, -34, 46, 26, -45, -9, 29, -30, -136, -119, -58, -15, 7, 5, -9, 111, -40, 42, -57, -63, -31, -50, -76, 93, 1, 9, -7, -93, 62, -21, 18, -14, 69, -26, -16, -50, -34, 3]),
(1120240085, 5061, [10, 23, -86, 17, -13, -5, -26, -58, -76, -166, -22, -12, 73, 56, -43, 61, 23, -25, -46, 29, -40, -58, -76, 35, -71, -53, -37, 23, 48, 132, -91, -9, -9, 32, -14, 21, -102, 14, 17, 79, 74, 37, -42, 54, 15, 91]),
(734443489, 3069, [-10, 78, -19, -49, 56, -61, -25, 25, -52, -33, -4, 82, -2, -11, -33, 90, 28, 16, -53, -14, -94, 32, 8, 9, 16, -39, -34, 73, 51, 41, -43, 51, 13, -58, -26, 82, -63, -64, 40, 41, -70, 92, 49, -15, 8, -161, 36]),
(432416163, 2401, [-45, 9, -76, 34, -10, -39, -15, -34, -62, -46, 76, -58, 57, 62, 51, 4, -9, -10, -11, -36, 33, -69, 34, -71, -67, -69, -49, -21, 88, -54, -54, -4, -24, -75, 113, -43, 12, 22, 64, -25, 7, -25, 14, 17, 93, -20, 24]),
(181296613, 805, [-5, 43, 57, -28, -28, -13, 57, 10, 5, 25, -34, -10, 67, 21, 50, 32, 52, 105, -38, 0, 38, 7, 40, -2, -28, 8, 16, -164, 40, 17, 83, 35, -48, 40, 6, 79, 0, 0, -11, -22, 18, 19, -78, -95, -111, 16, -61]),
(1206217497, 5208, [63, 33, 14, 39, 6, 14, -52, 30, -66, 31, 139, -26, 29, 70, -50, -56, 70, 42, -62, 2, 50, 101, -57, -4, -19, 58, 61, 16, -43, -1, -89, 1, 51, -8, -70, 120, 9, -17, 3, -6, 2, -74, 92, -2, -49, 55, -25]),
(848321905, 4076, [-15, -20, 91, 10, -18, 30, 48, -22, 31, 46, 0, 23, 39, 24, 70, -56, 30, 2, 12, -98, 90, 11, 5, 17, -9, 7, 76, 47, 25, -39, -26, -3, -74, -57, -53, -58, -58, -14, 55, 75, -53, -65, 27, -74, -17, -7, -28, -46]),
(476170232, 2147, [-3, -61, -77, -35, -33, -60, 66, 26, 39, 30, -44, -33, -85, -19, 14, -3, 19, 1, 3, -70, -15, 36, 35, -72, 33, -54, 53, -29, 19, -10, 2, 73, -4, 64, 72, 87, 82, -35, -10, -33, 17, -27, -31, -45, 87, -11, -25, 57]),
(499180479, 2675, [-36, -14, -16, 23, 17, -20, -35, 36, -37, -6, -40, 39, -22, -33, -59, -26, 32, 26, -67, -40, 25, -5, 89, -29, 23, 67, -73, 78, 1, -35, -66, 18, 40, 67, 39, -2, 72, 5, 14, -51, 9, -35, -24, 45, -79, 67, -12, -118]),
(1070890948, 4441, [47, 33, -65, -76, -46, 23, -19, -44, -20, 21, -117, 22, -5, -64, 76, -61, 26, 0, -57, 59, 25, -12, 60, 3, -5, 7, 85, -69, 59, 26, 10, -10, -13, 66, -42, 4, -30, 35, -43, 61, -1, 86, 19, -14, -42, 1, -48, -6]),
(18446744072900166516, 13688, [-93, 46, 17, -21, 3, 39, -32, 57, -41, 6, 62, -9, -87, 34, 5, 33, -35, 81, -74, -21, 20, 45, 1, 57, -41, -37, 22, 31, -42, 45, 72, -27, -27, 15, -6, 13, 20, 35, -24, -1, -31, 20, -42, 0, -43, -52, 17, -62, -25]),
(1914649432, 11023, [20, -58, -21, 53, -17, -44, 10, 25, 10, 70, -45, 5, -53, 16, 9, -59, -34, 35, -27, -39, 11, -53, 52, 35, -3, -31, -1, 23, -44, 41, -30, -22, 79, -47, -68, 24, -13, -19, -22, 11, 77, 82, -3, 6, 46, -32, -41, 38, 86]),
(18446744072767119169, 15435, [40, 55, -23, -61, -3, 17, -71, 31, -12, -98, -36, -21, -27, -1, 8, 1, 66, 51, -21, -13, 72, -4, -38, -48, 27, -28, -23, 9, 73, 58, -34, -8, 41, 55, 10, 52, -24, -12, 32, -7, -11, 43, 27, -4, 8, 32, 106, 8, -160]),
(1047336802, 29281, [-6, 30, 16, -9, 57, 98, 17, 2, 40, -28, -34, -2, -27, 29, -51, 27, 14, -58, 18, -55, 12, -39, -27, -1, -16, -38, -6, 75, -22, -7, 64, 33, 42, 47, -18, -30, 23, -63, 72, 1, -96, -73, 7, -52, -20, -32, 16, -31, 3]),
(1964805150, 46051, [-53, 21, 8, 25, -48, 14, -56, 37, -25, 34, -59, 14, -37, 49, 15, -36, -4, -29, 70, -47, -30, 18, 29, 2, -4, 55, 68, -66, -19, -13, 30, 30, -10, 63, -9, 4, -8, -62, -3, 24, -13, 23, -10, 19, -53, 11, -81, 46, 1, 4]),
(18446744073173255963, 17048, [22, -38, 31, -16, -13, 10, -23, -14, -42, 69, -13, 4, 3, -5, -23, -25, 43, -10, 43, 25, 47, -36, 26, -73, -2, -90, 7, 23, 20, 16, 26, 18, 51, 36, 35, 34, 24, -24, 6, -19, -17, 36, 13, 16, 46, -43, -1, 69, -75, -33]),
(18446744071840940585, 33823, [43, -32, -53, -31, 21, 47, 13, 21, 65, 7, -46, 42, 31, -22, -4, 26, 21, 69, -20, -95, 7, -36, 67, -24, 30, 26, -7, -84, -27, -38, -31, 29, -33, -22, -19, 5, -43, 11, -4, -36, -2, 17, 0, -55, -5, -2, 35, 29, 14, -1]),
(1107158055, 31515, [46, 8, 23, -11, 62, -69, 74, 39, 9, -52, 5, 3, -23, 16, -24, -44, -43, -56, 32, -36, 11, -3, 8, -12, -35, 42, 55, -17, -88, -7, 3, -7, 5, -20, 37, -62, -6, 9, -18, 41, 9, 39, 17, -62, -6, 69, -18, 2, 43, 8]),
(18446744071654287476, 51785, [3, 31, -24, 61, -10, -50, -5, -10, -12, -19, -7, 21, 14, -44, -6, -77, -27, 45, -9, 32, -25, -15, -25, 18, 11, 63, -8, -35, 5, 56, 69, -3, 36, -16, 27, 37, -47, 40, -9, 29, 2, -23, -6, -18, 12, 30, -5, -59, 26, -66, 43]),
(1972486617, 226103, [48, 36, -16, -28, -85, -34, -8, -22, -5, 3, 43, -43, 29, -31, 0, 13, -55, 22, 7, 59, -25, -2, -47, 8, -26, 18, -10, -30, 16, -24, 2, -5, 19, -29, 54, -4, 46, 4, -19, -32, -42, -19, 27, 62, -20, -30, -33, 68, -21, 22, -16]),
(614409966, 72026, [-3, 33, -6, -10, 4, 18, 25, -9, -24, 44, -35, -30, 7, 74, -41, -54, 3, -11, -3, 17, 45, 10, -42, 13, 17, -5, 28, -26, -8, 30, 1, 7, -23, -10, -51, 9, 21, 2, 36, -25, 5, -25, 29, -9, 36, -93, -54, -60, 69, -6, 60]),
(876577294, 37368, [17, -82, -13, 61, 0, -49, 30, 22, -2, -36, -9, -60, 5, 57, -18, -18, -13, -42, -36, 65, 9, -24, -24, 51, -76, 14, 36, 21, 17, -46, 32, 19, 41, -1, 9, -16, -3, 45, -21, 8, 52, -24, -41, 6, -2, -17, 9, -40, 8, -47, 24]),
(18446744073011559084, 208504, [-45, 4, -14, -12, -43, -7, -12, 5, 8, 4, -21, 19, -21, 56, 24, 2, -45, 34, -18, 17, 27, 6, -24, -26, -8, -19, -4, 10, 30, 9, 62, -35, 68, 1, 16, -23, -2, -8, 29, 46, 18, 33, -16, -30, -4, 31, 26, 27, 66, -31, -72, 7]),
(1761856653, 129564, [15, -36, 44, 20, 19, 23, -10, 6, 27, 7, -11, -10, 28, -12, -3, -29, 8, -4, 14, -50, -19, 57, 1, 21, 43, 47, -16, -3, -48, -11, -75, 34, 13, 1, 33, 11, -52, 27, -29, -40, 54, -75, 0, 5, -35, -5, 43, 49, -7, 3, 5, 11]),
(658379707, 192901, [-38, -41, -13, 45, -5, -29, 1, 18, -5, 13, 12, 55, 10, -23, 20, 4, -9, -12, 36, -26, -16, -7, -11, -73, 69, -25, 18, -13, 16, -7, 2, 29, 1, 44, 15, 6, 25, -84, 34, -19, -31, -2, -44, -12, 60, 7, -46, -10, -24, 12, -39, 9]),
(18446744073384241095, 279396, [-39, 1, 2, -37, 6, -4, 4, 23, 50, 22, 0, -15, 15, -75, 59, 5, -23, 44, 13, -1, -5, 7, -31, -36, 17, 39, 1, -20, 11, -63, 29, 56, -35, 10, 5, 8, -72, -5, 40, -34, 19, 24, 18, -11, 17, -11, 4, 30, 8, -8, -45, -52]),
(1752216138, 270699, [-20, -25, 4, -13, -43, 19, -19, 46, 10, 25, 33, 19, 20, -7, 19, -30, 15, -21, 43, 16, -35, -19, 34, -20, -26, 41, -16, 30, -19, 3, 20, -62, 22, -16, -20, 0, 47, -10, 16, 53, 15, -45, 44, 1, 19, 0, -19, 56, 32, -29, -27, 19, -31]),
(1300998449, 953540, [0, 1, 26, -23, -6, 16, 2, -56, 13, -1, 8, 18, 17, -4, 17, -52, -50, 27, -31, 33, 10, -22, 7, 24, 64, 31, -18, 28, 21, -12, 28, 59, 23, 6, -18, -7, 0, -11, 23, -25, 7, -58, -7, 29, 51, -45, 23, -33, -21, -2, -6, -18, -11]),
(1722859416, 390466, [23, -23, -10, -41, 40, 5, -33, -5, 2, -15, 9, -23, 50, 11, -1, 0, -12, 6, 56, 15, 8, -32, 35, -9, 27, -14, 18, -30, 52, -34, 12, 43, -2, -7, -34, -48, 9, -66, 27, -4, 14, -29, -28, 30, -28, -19, -40, 5, 29, -25, 29, 13, -34]),
(1316156034, 773219, [11, -21, -3, -41, 33, -38, -7, 12, 64, -17, -36, -64, 8, 13, -25, -16, 28, 30, 1, -23, -22, 6, 10, -33, -40, -5, 9, -8, 19, -6, 0, -35, -13, -3, 32, 33, -23, -40, 9, 18, 18, -34, 22, 14, 9, 12, -5, 7, 59, -47, -27, 13, 58]),
(18446744072752597001, 3098472, [-35, -30, 29, 6, -64, 35, 22, -13, 8, 24, 21, 8, -9, -4, -14, -27, 23, -14, 6, -8, 7, 9, 43, 18, 37, -61, -3, 0, 17, 14, -55, 46, -11, -1, -35, 29, -7, -1, 14, 22, -23, 7, -4, 14, 18, 22, -30, 0, -19, 43, -15, 32, -7, -35]),
(374523997, 1889356, [15, -32, 3, -22, -8, 19, -6, -25, -10, 32, -13, -5, -13, 6, -33, -31, -23, -67, -29, 28, -3, -33, -4, 2, 33, -6, -4, 4, -16, -10, 19, 5, 93, -5, -31, -18, 22, -3, -20, -24, 22, 35, 20, 15, 37, 60, -4, 0, -20, -1, 16, -17, -16, -12]),
(606145505, 3152194, [-8, 13, -1, 54, -5, -34, 43, -22, -30, -20, 1, 38, -22, -41, -12, -31, -1, -18, 3, 16, -28, -18, 43, 33, 26, -1, -20, -25, 21, 14, 22, 18, 11, -39, -27, -25, -2, 13, 15, 58, 37, -20, 28, 32, 11, 31, -2, 8, -33, 15, 5, 0, -38, -30]),
(1547061044, 1726859, [-6, -22, -11, -19, 60, -12, 55, -3, -43, 34, 59, 24, -24, 5, -48, 24, 1, -7, 3, -4, -17, 4, -11, -12, -8, 16, -32, 27, -32, -40, -7, 31, -27, -9, 52, -11, -1, 15, 37, -1, -17, -20, 29, -11, -52, 14, -20, -5, -8, 27, -1, 17, -13, 42]),
(18446744072903559366, 2995372, [-32, 27, -4, 1, 11, 32, 15, -48, -11, 9, 6, -7, 3, 26, 13, -19, 57, 47, 9, 11, -24, -45, 5, 13, -40, 29, -3, 29, -32, -4, 23, -42, 9, 2, 17, 20, 5, 27, 23, -1, -3, 29, 11, -37, -7, -15, -3, 8, -2, -42, -5, 0, 0, -20, 29]),
(18446744071855478875, 1932964, [12, 19, -11, -35, 5, 13, -7, 15, -1, -14, -24, -16, -15, -1, -10, 13, 20, -18, 31, 6, -18, 46, 22, -26, -14, 23, 23, 9, 13, -43, -15, -6, 35, 22, -13, 17, -8, 11, 12, 14, -26, 25, 36, -14, 1, -15, -17, 41, 9, -46, -15, -37, -32, 35, -34]),
(18446744073397663744, 33731583, [8, -1, -34, -21, -3, -65, -3, 0, -11, -6, 16, 11, 17, 61, 10, 6, 19, -15, -9, -17, -18, -18, 18, 10, 3, 29, 16, 13, 34, 1, -20, 29, 28, -22, -6, 21, -28, 16, 10, -8, -12, -33, -32, 34, 0, -14, 19, 16, -63, 42, 2, -13, 11, 1, 19]),
(18446744071956932517, 7310048, [10, 27, 17, 21, -3, 25, 24, -10, -23, -14, 37, -3, 22, -30, 4, 54, 15, -8, 2, 3, -10, 8, -34, 24, -9, 9, -2, 9, -5, -22, -2, -3, -18, 11, 21, 1, -22, 31, -32, -18, -26, 19, 23, 4, -32, -62, -13, -27, -5, -4, -4, 18, 43, 53, -24]),
(18446744073704731536, 1569898, [-34, 18, 45, -12, 15, 2, 24, 44, 24, 45, -20, 13, -4, -26, -10, -22, 6, 13, 5, 6, 17, 23, -5, -51, -3, 14, 4, -3, -4, 7, -36, 24, -15, -22, 4, 1, -12, -12, 5, -35, -15, -1, 7, -21, -3, 5, -8, -44, -4, 19, -43, -16, -2, 41, 5, -4]),
(247920644, 10788874, [-14, 32, -46, 25, -54, -22, -8, 40, -28, 27, 11, -12, -7, 3, -13, 16, 24, 9, -32, 11, 7, 19, 23, -4, -22, 7, -18, -11, -28, 0, 22, 21, 25, -22, -4, -7, 16, 13, -18, -25, 8, 9, 13, 22, -28, -28, 43, -5, -21, 11, -7, 12, 12, -6, 9, 28]),
(1763239056, 9155622, [3, 9, -7, 22, -23, -18, -27, -18, -33, 2, -2, -25, 48, 9, -13, 8, -34, 4, 3, -26, 33, 1, 10, -1, -19, -21, 4, -6, -19, -4, 26, 13, 19, 3, 19, -12, -14, 16, 59, 17, -11, -12, -1, 6, 30, 3, 22, -11, -30, -24, 7, -13, 13, 24, -14, 8]),
(460798011, 11621664, [8, -4, -4, 17, 23, 2, -48, -7, -5, -31, 6, -1, 0, -16, 8, -15, 0, 47, -35, 27, 3, -3, 19, -17, 7, 25, -7, 10, 11, -17, -9, -12, -10, -23, 14, -22, -21, -16, 11, 23, -29, 7, 14, 21, 10, 2, -9, -24, -19, 20, -39, 1, -36, -1, -10, 26, 18]),
(18446744072527797638, 7628093, [-21, -1, 2, -4, 34, -15, 4, 8, 13, -15, 12, 6, -25, -9, 5, 1, 1, 15, 6, 47, 16, 3, -9, -20, 4, 40, 10, -6, 20, 22, -20, 17, -23, -45, 5, 14, -6, -28, 3, -28, -24, -9, -27, -10, -57, -1, -1, 31, -5, -42, -14, 23, -16, 15, 16, 6, -7]),
(425601430, 2766281, [-22, -19, -23, -19, 11, -23, 7, 28, 27, -2, 3, -33, 6, 15, -1, 11, 12, 17, -17, -1, -23, -11, 11, 0, -25, 26, -41, -17, -2, 8, -19, -10, 40, -24, -6, 1, 3, 16, -18, 44, 8, 3, 9, 21, 23, 10, -45, -11, 20, -14, -19, -5, -34, 15, -9, 11, -19]),
(18446744072562190216, 9764904, [3, 9, -7, 22, -23, -18, -27, -18, -33, 2, -2, -25, 48, 9, -13, 8, -34, 4, 3, -26, 33, 1, 10, -1, -19, -21, 4, -6, -19, -4, 26, 13, 19, 3, 19, -12, -14, 16, 59, 17, -11, -12, -1, 6, 30, 3, 22, -11, -30, -24, 7, -13, 13, 24, -14, 8, 0]),

# dim 57
(0, 11017662, [5, -2, 13, 11, 7, -9, 12, 25, -8, 6, 8, 11, -8, -7, 8, 0, -13, -6, -5, 2, -12, 9, 3, -24, -1, -1, -8, -14, 4, 9, -2, -19, -3, -11, 17, 4, -1, 2, 1, -9, 2, 1, 0, 7, 13, 0, -7, -4, 2, 13, -5, -1, 6, -16, 8, -3, 12, -2]),
(0, 20611885, [3, 17, -7, 4, -1, -21, 2, -11, -11, 9, -7, -4, 3, -4, -9, 15, 4, -4, -2, 5, -16, 7, 13, 13, 9, 3, 0, 9, -1, 2, 11, 2, 12, -5, -6, -12, -13, -3, -24, 7, 15, -7, -5, -4, -4, 13, -8, 0, 12, 8, -6, 4, 14, -3, -5, 14, -3, -6]),
(0, 37003531, [-8, -5, -2, 0, -4, 2, -17, -2, -13, 3, -8, 5, -12, 7, 13, 1, -6, -23, -15, -8, -3, 0, -1, -3, 21, 1, -5, 19, 1, 18, -18, -2, -4, 11, 4, -5, 7, 14, 5, 15, -2, 3, 2, -14, -1, -10, -20, 5, 13, 9, -8, -1, 10, -11, 17, 2, -11, -3]),

# dim 58
(18446744071716225898, 44426184, [-26, 12, -7, -38, 8, 7, 11, -4, -5, -11, -1, 40, 6, 11, -1, 0, -33, 6, -4, 17, -21, -12, 16, 2, 8, 7, -3, -20, -9, 14, -9, 22, 13, 27, -5, -7, -30, -10, 9, -14, -8, 0, -11, 8, -9, 32, -1, -5, -1, 0, 1, 19, 15, -15, 17, -26, -8, -12, -2]),

(0, math.inf, [0 for _ in range(56)]), # the computations for seed ?? in dim 55 and seed 0 in dim 57 have needed so long i aborted them
(0, math.inf, [0 for _ in range(58)])
]

cu = [(33, 45358439, 490, [90, 60, 175, 30, -106, 47, 57, -72, 35, -220, -169, 77, -29, -7, -57, 117, -41, 55, -31, -69, -6, 118, 27, 31, 17, 145, -162, 36, 94, 243, -297, -81, -69, -174, 63, 111, -230, 0, 90, -25, -8]),
(33, 63918284, 297, [94, 165, -121, -38, 71, 58, 104, -69, 16, 186, 130, 5, -123, 129, -75, 65, -8, 211, -105, 83, 97, -48, -66, 127, -139, 61, -165, 18, 108, 91, -135, -52, 98, 50, -217, 84, 0, -9, 140, -231, -199]),
(33, 22777556, 311, [-93, -96, 60, 25, 110, -37, -12, -109, -176, -83, 107, 98, -8, -229, -23, -24, -112, -28, -149, 39, 41, -109, 11, 114, 166, -214, -9, -49, -229, 214, 32, -13, 57, 33, 200, -37, 182, 141, -122, 105, 154]),
(33, 3409837, 184, [-36, 57, -124, 6, 78, 24, -40, -20, 45, 128, -111, -152, 167, -138, -7, 97, 78, 81, -14, 16, -116, -40, -198, -64, 114, -135, -77, -98, 18, -49, 171, 76, -184, 154, -162, 57, 44, 293, -67, -4, -100]),
(33, 36165543, 266, [-79, -96, -31, -101, 19, -30, 16, -77, 43, -79, 117, -77, 55, -48, 63, -86, 35, -17, -69, 74, 165, 43, 84, 34, -189, 1, -114, 76, 21, 238, -110, 66, 81, 81, 19, -124, 218, -116, -126, -83, -147, 79]),
(33, 28130626, 194, [61, -222, 119, 60, -150, 29, 4, 120, 32, -98, -29, -61, 62, 0, -14, -22, 68, 59, -34, -17, 66, -53, 208, 32, 72, -128, 45, 8, 124, -167, 18, 1, 92, 40, -133, 26, 133, 39, -118, -2, -36, 180]),
(33, 27382735, 240, [-13, 149, -12, 30, -13, -39, 175, -29, 16, -76, 49, 184, -100, 99, -64, -95, -28, -113, -224, 100, -50, 50, -79, -193, -47, 40, 148, 3, 118, -43, 109, 41, 36, -135, 80, -24, -93, -105, -130, -135, 24, 34]),
(33, 131086296, 644, [230, -45, -4, 35, -26, 100, 4, 42, 7, -176, 48, -105, 70, 129, -67, 4, 74, 98, 46, -98, 66, -153, 124, 209, 43, -110, 2, -43, -81, -50, 51, 56, -161, 147, 62, 58, -231, -83, 9, -27, -181, 108]),
(36, 54664660, 708, [26, -41, -151, 12, 98, -25, -148, -13, 51, -166, 61, 7, 51, 33, -63, 57, -152, 69, -18, -35, 55, -116, -118, -100, -86, -62, -58, 97, 62, 114, 105, -15, -200, 77, 75, 106, 61, 80, 51, 21, -6, -85, -204]),
(36, 23210462, 532, [-41, 78, -80, -19, 61, -5, 0, -25, 36, -124, -100, -81, 133, -62, 113, -2, -198, 31, 13, 52, -67, -114, -16, -10, -34, 12, 72, 118, 10, 86, 87, -55, 55, 53, -77, 54, 176, 20, 37, -28, -72, -194, -75]),
(36, 63289210, 541, [-112, 26, -7, -50, 47, -143, 83, 220, -20, 164, -9, -35, -24, 24, 12, 67, 76, 40, 105, -17, -74, 90, -21, 105, 41, 20, -71, -41, -137, -102, 34, -19, -87, -7, -201, 66, 67, 16, 18, 76, 46, 0, -263]),
(36, 78704092, 1237, [43, 26, -38, 86, -31, -77, 15, 29, -13, 154, 14, 9, 95, 138, 35, -75, -118, -58, -105, 44, 162, -51, -112, -22, -141, -5, -44, 19, -59, -154, -69, 177, 2, -100, -63, -89, -86, -37, 29, 116, 94, -11, 198]),
(36, 116727254, 641, [9, -24, 49, -21, 50, -110, 4, 46, 28, 72, -43, 67, -135, -60, 137, 146, 11, -116, -82, 1, 1, -56, 9, 27, 14, 8, 3, -178, -77, -36, 45, 110, 122, -17, 22, -80, -57, 21, 57, 6, 25, 37, 25, -188]),
(36, 247642295, 1016, [-70, 19, -60, 10, -70, 106, 116, 64, -42, -61, -73, -27, 43, 101, 123, 81, -65, -43, 58, 4, -26, -49, 83, -89, -75, -6, 47, -176, 73, 8, -15, 72, -52, 53, 20, -15, 24, 132, 1, -4, 68, -208, 20, -32]),
(36, 133849013, 867, [-2, -112, -35, -73, -92, -63, -23, 150, 0, -18, -58, 120, 28, 30, -93, -72, 119, -56, -103, -14, 28, 27, 73, -65, 39, 83, 55, -77, 119, -37, -27, -88, -102, -141, 41, 52, -51, -93, -51, -64, 85, 126, -17, 76]),
(36, 189370699, 1110, [-21, 32, 0, -44, 114, 23, -62, 51, -128, 67, -151, -155, 9, -97, 18, -31, 37, 10, 21, -128, -75, 108, 54, 63, 21, -50, -72, 28, 97, -77, -31, 145, 17, -106, -35, 111, 39, 83, -37, 59, -89, 46, -7, -29]),
(36, 234983575, 763, [-62, 61, -43, 29, -54, -117, 19, 36, 31, -38, 8, 75, 7, 101, 71, 7, 17, -6, 62, -142, 68, 148, 36, -43, -41, -48, 17, -7, 12, -1, -106, 6, 70, -17, -27, -18, 75, -62, 59, -160, 31, -63, 0, 44, 118]),
(36, 1247891676, 2712, [13, -12, -104, -10, 34, 22, -64, 6, 18, 56, 40, -15, -6, -49, 18, -95, -57, 113, -24, -13, 121, 24, -85, -103, -93, -142, 2, -66, -61, 24, 90, -45, -56, 38, 31, -12, 93, -9, -2, -40, 62, 120, -74, 101, 106]),
(36, 400066441, 1050, [-3, -67, 44, 17, -44, 10, 68, 41, 24, 10, -10, 17, -64, -91, 11, 51, -1, -18, 11, -116, -109, 34, -79, -26, -32, -8, 64, -12, 47, 33, -1, -90, 126, 1, 31, 145, -114, 132, 34, 19, -134, -67, 70, -17, 152]),
(36, 114557137, 513, [41, 11, -17, 51, 104, 40, 125, 86, 1, -136, -47, -46, 16, 4, -13, 86, 62, 22, 53, -54, -96, 11, -50, 111, 106, 20, 27, 23, 6, -79, -15, 36, 16, -13, 10, -56, -115, 93, -106, -59, 0, -38, -95, -99, -34]),
(39, 617948999, 5243, [25, -1, -37, 41, 47, -63, 101, -5, 17, -35, 40, -18, 12, 46, -89, 50, 67, -17, -95, -78, -23, 114, -88, 52, -99, 4, 49, -75, 107, 21, -11, -82, 105, 86, 19, 19, -32, -64, -45, -4, 11, -103, 14, -14, -22, -22]),
(39, 1789428464, 6054, [4, 5, -71, 25, 29, -36, -13, -44, -44, -86, 44, -44, -31, 42, -60, 78, 93, -96, 56, -18, -103, 24, -5, 53, -74, 131, 19, 5, -16, 17, -61, 57, -43, -126, -16, -18, -11, 1, 13, -29, -122, 72, -1, 59, 22, -30]),
(39, 918944080, 8283, [-35, 146, 95, 94, -21, -25, 54, -22, -14, -34, 46, 26, -45, -9, 29, -30, -136, -119, -58, -15, 7, 5, -9, 111, -40, 42, -57, -63, -31, -50, -76, 93, 1, 9, -7, -93, 62, -21, 18, -14, 69, -26, -16, -50, -34, 3]),
(39, 2642326483, 10116, [66, 4, -61, -19, -12, 84, 9, -25, -22, 28, 47, -47, 53, -59, 52, -38, -12, 9, 76, 17, -45, 2, -99, -139, 19, 89, -9, -92, -9, -72, 108, -10, -15, 2, -31, -62, 92, 108, -60, 43, -47, -35, -72, 78, -24, 112]),
(39, 1421702805, 5814, [-31, -27, 1, -13, -36, -70, -50, 12, -40, 32, 88, 20, 29, 93, 94, 19, -40, -5, 59, 2, 19, -7, -62, -41, -12, 32, -31, -5, -51, -4, 51, 35, 74, 14, -21, 105, -72, -4, 112, 61, 39, -20, -13, -70, -145, 55, 143]),
(39, 1432740517, 3781, [-45, 9, -76, 34, -10, -39, -15, -34, -62, -46, 76, -58, 57, 62, 51, 4, -9, -10, -11, -36, 33, -69, 34, -71, -67, -69, -49, -21, 88, -54, -54, -4, -24, -75, 113, -43, 12, 22, 64, -25, 7, -25, 14, 17, 93, -20, 24]),
(39, 353349472, 1984, [-5, 43, 57, -28, -28, -13, 57, 10, 5, 25, -34, -10, 67, 21, 50, 32, 52, 105, -38, 0, 38, 7, 40, -2, -28, 8, 16, -164, 40, 17, 83, 35, -48, 40, 6, 79, 0, 0, -11, -22, 18, 19, -78, -95, -111, 16, -61]),
(39, 2368306389, 9485, [86, 89, 11, -8, -11, -99, -59, -12, 28, 47, 16, -9, -28, 100, -62, 15, 30, -67, -52, -53, 34, -59, 103, -19, -40, 52, -73, -13, 6, 104, -94, 44, -70, -67, 40, 61, -56, -7, 57, -11, 33, 53, 24, -30, -4, 28, 93]),
(39, 833937379, 2545, [10, 149, -21, 22, -4, 24, 11, -17, 0, 19, -26, -53, 66, -36, -43, -8, 24, -58, 8, -6, -20, 13, -8, -3, 70, -40, 22, 79, 31, -9, -47, -45, -39, -3, -13, 4, -7, 29, 19, 53, -49, 2, 68, 72, -58, 110, 90, 54]),
(39, 843898665, 2271, [1, -93, -19, -1, -2, 13, -7, -80, 7, -35, -67, -57, 30, 45, 13, 8, 54, 11, -12, 6, -44, -6, -96, 43, -15, -123, 12, 55, 45, -10, -56, 52, -13, -28, -1, -25, 4, -15, -40, 8, 51, 88, 57, 6, 87, -47, 10, -48]),
(39, 1059433648, 2312, [-36, -14, -16, 23, 17, -20, -35, 36, -37, -6, -40, 39, -22, -33, -59, -26, 32, 26, -67, -40, 25, -5, 89, -29, 23, 67, -73, 78, 1, -35, -66, 18, 40, 67, 39, -2, 72, 5, 14, -51, 9, -35, -24, 45, -79, 67, -12, -118]),
(39, 1693847396, 5256, [-49, 87, 67, -41, -21, -12, -99, -71, 11, 56, 10, -44, 18, 45, 49, 36, -50, -34, -20, -18, -16, -13, -5, 17, 5, 15, 62, -37, 15, 94, -87, -18, -48, 96, 16, 61, 23, 9, -22, -34, 79, 19, 1, -50, -41, 4, -43, 40]),
(42, 6594127493, 40694, [22, 12, 7, 22, 24, -17, -12, 95, -7, 77, 11, 38, 3, -25, -30, 8, -60, 8, -57, 38, 22, 15, 23, -59, 27, 18, 2, -15, 10, -60, -14, -22, -35, 6, -23, -17, 59, -81, 31, 23, 116, 0, 6, -55, -86, 56, 35, 1, 61]),
(42, 7623003275, 25367, [20, -58, -21, 53, -17, -44, 10, 25, 10, 70, -45, 5, -53, 16, 9, -59, -34, 35, -27, -39, 11, -53, 52, 35, -3, -31, -1, 23, -44, 41, -30, -22, 79, -47, -68, 24, -13, -19, -22, 11, 77, 82, -3, 6, 46, -32, -41, 38, 86]),
(42, 9835415450, 40763, [-34, 17, 20, 28, -69, -23, -4, 46, 31, 13, 17, 35, 24, 33, -41, 79, -26, 68, 35, -36, 23, 24, 4, 12, 61, -84, -18, -46, -78, -70, -65, -23, -20, -13, 36, 32, 65, 25, 13, 30, 28, 20, 43, -16, -7, -11, 21, -100, -60]),
(42, 10517181187, 56315, [-6, 30, 16, -9, 57, 98, 17, 2, 40, -28, -34, -2, -27, 29, -51, 27, 14, -58, 18, -55, 12, -39, -27, -1, -16, -38, -6, 75, -22, -7, 64, 33, 42, 47, -18, -30, 23, -63, 72, 1, -96, -73, 7, -52, -20, -32, 16, -31, 3]),
(42, 7454959321, 27012, [41, 60, -69, 17, -7, 73, -37, -49, -8, -40, -93, -21, 14, -10, -47, 6, 63, -17, 36, 12, -32, -22, 20, -70, 2, -24, -8, 51, 11, 38, -31, 5, -21, -35, 43, 4, -1, -77, 19, -7, -2, 20, 3, -31, 11, 27, -26, -17, -20, 62]),
(42, 10858310274, 25921, [22, -38, 31, -16, -13, 10, -23, -14, -42, 69, -13, 4, 3, -5, -23, -25, 43, -10, 43, 25, 47, -36, 26, -73, -2, -90, 7, 23, 20, 16, 26, 18, 51, 36, 35, 34, 24, -24, 6, -19, -17, 36, 13, 16, 46, -43, -1, 69, -75, -33]),
(42, 11707378836, 31648, [43, -32, -53, -31, 21, 47, 13, 21, 65, 7, -46, 42, 31, -22, -4, 26, 21, 69, -20, -95, 7, -36, 67, -24, 30, 26, -7, -84, -27, -38, -31, 29, -33, -22, -19, 5, -43, 11, -4, -36, -2, 17, 0, -55, -5, -2, 35, 29, 14, -1]),
(42, 17403372309, 50751, [46, 8, 23, -11, 62, -69, 74, 39, 9, -52, 5, 3, -23, 16, -24, -44, -43, -56, 32, -36, 11, -3, 8, -12, -35, 42, 55, -17, -88, -7, 3, -7, 5, -20, 37, -62, -6, 9, -18, 41, 9, 39, 17, -62, -6, 69, -18, 2, 43, 8]),
(42, 7852358731, 16606, [-19, -15, -19, 30, 2, -6, 12, -19, -7, 43, -15, 40, -4, -14, 21, 5, 43, -15, 4, -4, 49, 25, -59, -32, -7, 4, 27, 8, 15, 8, -17, 65, 27, 21, -38, -61, -18, 21, -32, -39, -54, 8, -19, -36, 4, 10, 49, -94, -8, 30, -90]),
(42, 27641455462, 60045, [48, 36, -16, -28, -85, -34, -8, -22, -5, 3, 43, -43, 29, -31, 0, 13, -55, 22, 7, 59, -25, -2, -47, 8, -26, 18, -10, -30, 16, -24, 2, -5, 19, -29, 54, -4, 46, 4, -19, -32, -42, -19, 27, 62, -20, -30, -33, 68, -21, 22, -16]),
(42, 14913973179, 32776, [-3, 33, -6, -10, 4, 18, 25, -9, -24, 44, -35, -30, 7, 74, -41, -54, 3, -11, -3, 17, 45, 10, -42, 13, 17, -5, 28, -26, -8, 30, 1, 7, -23, -10, -51, 9, 21, 2, 36, -25, 5, -25, 29, -9, 36, -93, -54, -60, 69, -6, 60]),
(42, 11117990579, 40848, [-15, 6, -45, -35, 3, -25, -77, -2, -2, -12, -26, 29, 31, 47, 19, 49, 59, -6, 36, -15, -10, 49, 65, -32, 34, -13, 2, 34, -14, -18, 7, -4, 4, -56, 35, 20, -29, -44, -81, 16, 44, 12, -68, -41, 2, -25, 16, 22, 35, 38, -9]),
(45, 86513157319, 354691, [-45, 4, -14, -12, -43, -7, -12, 5, 8, 4, -21, 19, -21, 56, 24, 2, -45, 34, -18, 17, 27, 6, -24, -26, -8, -19, -4, 10, 30, 9, 62, -35, 68, 1, 16, -23, -2, -8, 29, 46, 18, 33, -16, -30, -4, 31, 26, 27, 66, -31, -72, 7]),
(45, 90509591183, 327921, [10, -72, -67, 26, 40, -14, 3, -40, -4, 46, 0, 14, 7, 28, -28, 51, 45, 5, 7, -57, -28, -13, 15, -19, 33, 32, -27, -34, -5, -36, 25, 33, 3, 37, 37, -10, -7, -16, -23, -3, -11, 37, -37, 46, -65, 8, 5, 30, -25, -9, 8, 33]),
(45, 54607239729, 213158, [-33, -10, 29, -13, 13, -42, 8, 28, 11, 35, -67, 52, -7, -11, 4, -59, 29, 35, -19, -54, 12, 45, 26, 24, 9, -28, 31, -6, 26, -22, 15, 13, -13, 22, 16, -40, -6, 6, -17, -43, -68, -29, -23, 8, -8, -9, 52, 21, 18, 35, 58, 78]),
(45, 62652469396, 207247, [-39, 1, 2, -37, 6, -4, 4, 23, 50, 22, 0, -15, 15, -75, 59, 5, -23, 44, 13, -1, -5, 7, -31, -36, 17, 39, 1, -20, 11, -63, 29, 56, -35, 10, 5, 8, -72, -5, 40, -34, 19, 24, 18, -11, 17, -11, 4, 30, 8, -8, -45, -52]),
(45, 40926549774, 185955, [-9, 3, -31, -20, 9, 34, -13, 17, -7, -29, -55, 45, -14, 11, -22, 22, -7, -5, -14, 5, 20, -38, 28, -3, 33, 33, 10, 8, 28, -51, -26, -14, -29, -12, -14, 45, 38, -4, 30, -35, 13, -21, -66, 6, -18, 3, -10, 17, -15, 77, 27, -52, -66]),
(45, 104635930423, 266989, [0, 1, 26, -23, -6, 16, 2, -56, 13, -1, 8, 18, 17, -4, 17, -52, -50, 27, -31, 33, 10, -22, 7, 24, 64, 31, -18, 28, 21, -12, 28, 59, 23, 6, -18, -7, 0, -11, 23, -25, 7, -58, -7, 29, 51, -45, 23, -33, -21, -2, -6, -18, -11]),
(45, 78452103712, 268855, [25, 51, -43, 21, -39, -25, -15, 28, -17, -33, -1, -1, -4, 25, 30, 45, 30, -18, 22, 24, 5, -23, 14, 2, -24, -11, -33, -17, -70, 32, -28, -23, 48, -38, -20, 20, -12, 21, -41, -6, 26, 43, -2, -37, -36, -12, 13, 6, -3, 26, -21, 23, 7]),
(45, 101781272853, 217620, [11, -21, -3, -41, 33, -38, -7, 12, 64, -17, -36, -64, 8, 13, -25, -16, 28, 30, 1, -23, -22, 6, 10, -33, -40, -5, 9, -8, 19, -6, 0, -35, -13, -3, 32, 33, -23, -40, 9, 18, 18, -34, 22, 14, 9, 12, -5, 7, 59, -47, -27, 13, 58]),
(45, 161631036846, 419692, [-3, -9, -33, -4, -28, 11, -16, 19, -32, 21, -24, -18, -1, 18, 12, 66, 31, -23, 12, 10, 26, -39, 33, -14, 16, -41, -16, 26, -38, 28, -11, 9, 29, -39, 4, 19, 45, -4, -21, 6, 6, 20, 27, -42, -51, 16, 12, -14, 43, 18, -21, 18, 3, 5]),
(45, 203065458833, 412843, [15, -32, 3, -22, -8, 19, -6, -25, -10, 32, -13, -5, -13, 6, -33, -31, -23, -67, -29, 28, -3, -33, -4, 2, 33, -6, -4, 4, -16, -10, 19, 5, 93, -5, -31, -18, 22, -3, -20, -24, 22, 35, 20, 15, 37, 60, -4, 0, -20, -1, 16, -17, -16, -12]),
(45, 177781479142, 336940, [15, -25, 6, -5, 9, 25, -3, 57, -5, 1, -3, -12, 22, 10, -3, -26, 45, -16, 4, -7, -5, -8, -21, -27, -10, -15, -31, -46, 40, 3, -1, 36, 25, -32, 4, -10, 33, 14, -6, 4, 42, -43, 5, 39, -24, -9, 29, -59, 1, -37, -11, 20, 11, -64]),
(45, 96613468967, 217241, [23, 31, -17, -28, -25, 1, 1, 15, 24, 25, 47, -27, 56, -55, 26, 14, -11, 11, -27, -10, 41, -2, -19, -31, 1, -51, -34, -2, 4, 37, -39, 22, 10, 10, 31, -4, -4, 16, 24, -23, -45, -9, 43, -5, 18, 2, -25, 1, -10, -1, -21, -4, 39, -49]),
(48, 270571650029, 995359, [-32, 27, -4, 1, 11, 32, 15, -48, -11, 9, 6, -7, 3, 26, 13, -19, 57, 47, 9, 11, -24, -45, 5, 13, -40, 29, -3, 29, -32, -4, 23, -42, 9, 2, 17, 20, 5, 27, 23, -1, -3, 29, 11, -37, -7, -15, -3, 8, -2, -42, -5, 0, 0, -20, 29]),
(48, 273687711834, 687316, [24, 27, -49, 16, 5, -13, -6, -55, 37, 12, 16, 20, -11, 11, -19, -1, 16, 5, 12, 11, 11, -21, -36, -19, 13, -12, 23, 46, 14, -7, 0, -1, 6, 1, -51, -42, -24, -22, -2, 27, 13, -24, 14, -22, 28, 15, -9, 7, -7, 4, -7, -13, 13, -26, 68]),
(48, 2456997177983, 10349578, [8, -1, -34, -21, -3, -65, -3, 0, -11, -6, 16, 11, 17, 61, 10, 6, 19, -15, -9, -17, -18, -18, 18, 10, 3, 29, 16, 13, 34, 1, -20, 29, 28, -22, -6, 21, -28, 16, 10, -8, -12, -33, -32, 34, 0, -14, 19, 16, -63, 42, 2, -13, 11, 1, 19]),
(48, 880312026436, 3008395, [-13, -39, 7, 11, -32, -37, -24, 50, -48, 23, 5, -19, 7, -12, -10, 20, -14, 28, -6, 11, -11, -9, 7, -21, 21, 48, -1, 23, 37, -19, -32, 3, -37, 49, 11, -17, 18, 5, -30, 1, -9, -5, 30, 11, 29, 1, 6, 17, -18, -5, 11, 2, 29, 19, 35]),
(48, 152708632535, 437463, [-11, -23, 5, 14, 14, -33, -12, -1, -27, 4, -22, -5, 5, 31, 6, 23, -24, 18, 5, -21, -7, -22, 29, -3, -17, 18, -13, -17, -8, 5, -38, -4, -59, -20, 5, 28, 5, -7, 15, -6, 36, 29, 33, -30, 23, -31, -3, -9, -28, 39, -9, 12, 15, -9, 26, -48]),
(48, 1167235896699, 2587338, [-14, 32, -46, 25, -54, -22, -8, 40, -28, 27, 11, -12, -7, 3, -13, 16, 24, 9, -32, 11, 7, 19, 23, -4, -22, 7, -18, -11, -28, 0, 22, 21, 25, -22, -4, -7, 16, 13, -18, -25, 8, 9, 13, 22, -28, -28, 43, -5, -21, 11, -7, 12, 12, -6, 9, 28]),
(48, 4904217956532, 11283949, [-11, 15, 8, 18, 17, 16, 6, 21, 6, -31, 21, -1, 31, -7, 16, 10, 6, -8, -41, -29, -16, -41, -8, 16, -14, 17, -10, -5, -38, 1, -18, 6, -32, 11, 16, 27, -3, -11, 24, -5, -15, -43, -59, 10, -17, -3, -21, 10, 2, 0, -28, -20, -6, 22, 48, -25]),
(48, 609496580524, 1228309, [3, 9, -7, 22, -23, -18, -27, -18, -33, 2, -2, -25, 48, 9, -13, 8, -34, 4, 3, -26, 33, 1, 10, -1, -19, -21, 4, -6, -19, -4, 26, 13, 19, 3, 19, -12, -14, 16, 59, 17, -11, -12, -1, 6, 30, 3, 22, -11, -30, -24, 7, -13, 13, 24, -14, 8]),
(48, 374341080810, 729693, [8, -4, -4, 17, 23, 2, -48, -7, -5, -31, 6, -1, 0, -16, 8, -15, 0, 47, -35, 27, 3, -3, 19, -17, 7, 25, -7, 10, 11, -17, -9, -12, -10, -23, 14, -22, -21, -16, 11, 23, -29, 7, 14, 21, 10, 2, -9, -24, -19, 20, -39, 1, -36, -1, -10, 26, 18]),
(48, 496989520140, 1190493, [-21, -1, 2, -4, 34, -15, 4, 8, 13, -15, 12, 6, -25, -9, 5, 1, 1, 15, 6, 47, 16, 3, -9, -20, 4, 40, 10, -6, 20, 22, -20, 17, -23, -45, 5, 14, -6, -28, 3, -28, -24, -9, -27, -10, -57, -1, -1, 31, -5, -42, -14, 23, -16, 15, 16, 6, -7]),
(48, 388274052474, 746792, [8, -11, -28, -1, 46, -30, -25, 37, 26, -14, -17, -5, -10, 6, -11, 1, -29, 23, -13, 37, 2, -11, 8, -9, 29, 3, 2, 8, -5, -11, 14, -22, 18, 46, -26, 16, -20, 17, 11, -4, -7, 40, 6, 2, 11, -6, 8, -10, 17, -22, -8, -17, 5, 10, 17, -3, -38]),
(48, 567626277799, 1092544, [3, 9, -7, 22, -23, -18, -27, -18, -33, 2, -2, -25, 48, 9, -13, 8, -34, 4, 3, -26, 33, 1, 10, -1, -19, -21, 4, -6, -19, -4, 26, 13, 19, 3, 19, -12, -14, 16, 59, 17, -11, -12, -1, 6, 30, 3, 22, -11, -30, -24, 7, -13, 13, 24, -14, 8, 0]),
(51, 1430593615835, 7036961, [25, 14, 17, -28, 2, 17, -1, 10, 25, -2, -22, 4, -27, -5, -3, 36, -5, 46, -8, -15, -21, 4, -15, -4, -16, 19, -1, 12, -17, 6, -18, -26, -13, -13, -20, 25, -34, -6, 2, 49, -2, 19, 13, 2, -28, 13, 12, -11, 11, -18, -17, 15, 5, 22, -5, -10, -28, 19]),
(51, 1430835419493, 7285869, [25, 14, 17, -28, 2, 17, -1, 10, 25, -2, -22, 4, -27, -5, -3, 36, -5, 46, -8, -15, -21, 4, -15, -4, -16, 19, -1, 12, -17, 6, -18, -26, -13, -13, -20, 25, -34, -6, 2, 49, -2, 19, 13, 2, -28, 13, 12, -11, 11, -18, -17, 15, 5, 22, -5, -10, -28, 19]),
(51, 1755062423580, 6783648, [21, -36, 11, 3, 22, -15, -29, 3, 14, 10, 26, -11, -20, 21, -14, 40, -30, 1, 10, 5, 21, 19, -6, -35, 9, -19, 3, 7, 23, -2, -16, 0, 34, 2, -18, -24, -4, -17, -19, 4, -15, -22, -10, 19, -21, -25, -4, -22, 3, 15, 30, -2, 10, 9, 11, 9, 27, -38]),
(51, 5501032909543, 13558267, [29, -18, -4, 10, 11, -22, -43, 14, 9, 5, -34, -18, 18, -15, -8, -16, 9, -2, -6, 14, -17, 16, 15, 10, -17, 0, -5, -22, -28, -18, 19, -2, -7, -20, 13, -31, 15, 10, 21, 21, -16, 17, -10, 21, 24, -32, -7, 4, 10, -11, 18, 11, -3, -8, -26, 16, 11, -27]),
(51, 2675774077084, 6181855, [-11, -8, -2, -3, 21, 13, -2, 31, -16, 5, -16, -38, -12, 5, -23, 14, -35, -9, 18, -33, -16, 18, 2, 13, -10, 0, -21, 1, -4, 9, 7, -10, -4, 4, 5, 16, -5, -29, -7, 19, -9, 1, -19, -29, 27, 2, 3, -4, 0, -15, 25, 19, 8, -32, -1, 9, 17, -2, -43]),
(51, 6757739592073, 18500655, [-13, -8, 29, -23, -16, 7, 9, 15, 18, 2, 4, 9, 11, -2, -21, -24, -16, 7, -6, 0, 4, 7, -1, -27, 8, 15, 13, 5, 8, 4, 42, -11, -29, -35, 33, -34, -15, 24, 2, 1, 29, -11, 0, 28, -7, 31, 0, -29, 17, 14, 10, -8, -7, -16, 8, 11, 4, -2, -9]),
(51, 1997883658702, 5242285, [23, -25, 17, 18, -41, 6, 15, 27, 10, -11, -13, -20, 2, -20, -25, 12, 11, -2, 5, -33, 17, -12, 1, -8, 14, 32, -6, -13, 19, 0, -11, 3, 1, 24, 26, 7, 4, 9, -3, -7, -17, -27, -1, -21, 3, -4, 0, -11, 17, 18, -11, -32, 24, 13, 15, 0, 7, 5, -34]),

(51, 0, math.inf, [0 for _ in range(59)])
]

def median(l):
    x = list(sorted(l))
    if len(x) % 2 == 0:
        return (x[len(x)//2 - 1] + x[len(x)//2]) / 2
    else:
        return x[len(x)//2]

def collect_points(it):
    dim_map = {}
    for (t, _, v) in it:
        if len(v) - 1 not in dim_map:
            dim_map[len(v) - 1] = []
        dim_map[len(v) - 1] += [t]
    for key in dim_map:
        if len(dim_map[key]) != 4:
            print("Missing entry for dim " + str(key))
    return dim_map

def log_curvefit(x, y):
    lx = np.array(x)
    ly = np.log(np.array(y))
    valid = np.logical_and(np.logical_not(np.isnan(ly)), np.logical_not(np.isinf(ly)))
    lx = lx[valid]
    ly = ly[valid]
    A = np.concatenate((np.ones((1, len(lx))), np.reshape(lx, (1, -1))), axis=0)
    res = np.linalg.lstsq(A.T, ly)[0]
    return (math.e**res[0], res[1])

def ms_to_min(x):
    return x / 1000 / 60

def print_fit_curve(it, prefix, start = math.nan, end = math.nan):
    dim_t_map = collect_points(it)
    if math.isnan(start):
        start = min(dim_t_map)
    if math.isnan(end):
        end = max(dim_t_map)

    xy = [(d, t) for d in dim_t_map for t in dim_t_map[d] if d >= start and d <= end]
    (a, b) = log_curvefit([x for (x, _) in xy], [y for (_, y) in xy])

    for d in range((end - start) * 10):
        print(str(d/10 + start) + " " + str(ms_to_min(a*math.e**(b*(d/10 + start)))) + " " + prefix + "fit")

def print_data(it, prefix):
    dim_t_map = collect_points(it)
    for d in dim_t_map:
        t = dim_t_map[d]
        for i in range(len(t)):
            print(str(d) + " " + str(ms_to_min(t[i])) + " " + prefix + str(i))
        print(str(d) + " " + str(ms_to_min(median(t))) + " " + prefix + "med")
        print(str(d) + " " + str(ms_to_min(sum(t)/len(t))) + " " + prefix + "avg")

print_data([(t, n, v) for (_, n, t, v) in cu], "cu-")
print_fit_curve([(t, n, v) for (_, n, t, v) in cu], "cu-", end=57)
print_data([(t, n, v) for (n, t, v) in mt], "mt-")
print_fit_curve([(t, n, v) for (n, t, v) in mt], "mt-", end=57)