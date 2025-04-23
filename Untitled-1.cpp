#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>

using namespace std;

struct Room {
    int capacity;
    int rate;
    string status = "vacant";
    string guestID = "";
    string availableTime = "";
};

map<int, Room> hotel;
vector<int> WEEKENDS = {5, 6};
const double WEEKEND_RATE_MULTIPLIER = 1.2;

string addMinutesToTime(string timeStr, int minutes) {
    struct tm tm = {};
    istringstream ss(timeStr);
    ss >> get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    tm.tm_min += minutes;
    mktime(&tm);
    ostringstream output;
    output << put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return output.str();
}

void checkIn(string requestTime, int roomNumber, int adults, int children, int days, string guestID) {
    if (hotel[roomNumber].status == "occupied") {
        cout << requestTime << " check-in error: " << roomNumber << " is occupied." << endl;
        return;
    }
    if (hotel[roomNumber].status == "cleaning" && requestTime < hotel[roomNumber].availableTime) {
        cout << requestTime << " check-in error: " << roomNumber << " is being cleaned." << endl;
        return;
    }
    if (adults + children > hotel[roomNumber].capacity) {
        cout << requestTime << " check-in error: " << roomNumber << " cannot accommodate " << guestID << "." << endl;
        return;
    }
    hotel[roomNumber].status = "occupied";
    hotel[roomNumber].guestID = guestID;
    cout << requestTime << " check-in " << guestID << " successfully checked in to " << roomNumber << "." << endl;
}

void checkOut(string requestTime, string guestID, int roomNumber, int stayDuration, int adults, int children) {
    if (hotel[roomNumber].guestID != guestID) {
        cout << requestTime << " check-out error: " << guestID << " is not in " << roomNumber << "." << endl;
        return;
    }
    
    time_t now = time(0);
    struct tm *ltm = localtime(&now);
    int weekday = ltm->tm_wday;
    double dailyRate = hotel[roomNumber].rate * (find(WEEKENDS.begin(), WEEKENDS.end(), weekday) != WEEKENDS.end() ? WEEKEND_RATE_MULTIPLIER : 1.0);
    double totalCharge = stayDuration * dailyRate * (adults + 0.8 * children);
    
    cout << requestTime << " check-out " << guestID << " has to pay " << fixed << setprecision(2) << totalCharge << " to leave " << roomNumber << "." << endl;
    
    string cleaningEndTime = addMinutesToTime(requestTime, 180);
    cout << "cleaning of " << roomNumber << " will be completed at " << cleaningEndTime << "." << endl;
    hotel[roomNumber].status = "cleaning";
    hotel[roomNumber].guestID = "";
    hotel[roomNumber].availableTime = cleaningEndTime;
}

int main() {
    int N;
    cin >> N;
    for (int i = 0; i < N; i++) {
        int roomNumber, capacity, rate;
        cin >> roomNumber >> capacity >> rate;
        hotel[roomNumber] = {capacity, rate, "vacant", "", ""};
    }
    
    string command;
    cin.ignore();
    while (getline(cin, command)) {
        stringstream ss(command);
        string requestTime, action;
        ss >> requestTime >> action;
        
        if (action == "check-in") {
            int roomNumber, adults, children, days;
            string guestID;
            ss >> roomNumber >> adults >> children >> days >> guestID;
            checkIn(requestTime, roomNumber, adults, children, days, guestID);
        } else if (action == "check-out") {
            string guestID;
            int roomNumber, stayDuration, adults, children;
            ss >> guestID >> roomNumber >> stayDuration >> adults >> children;
            checkOut(requestTime, guestID, roomNumber, stayDuration, adults, children);
        }
    }
    return 0;
}
