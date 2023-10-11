#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>  // for sleep()
#include <cstdint>  // for int16_t

const float SIMULATION_WIDTH = 8.0f;
const float SIMULATION_HEIGHT = 8.0f;
const float SMOOTHING_RADIUS = 0.2f;
const int NUM_PARTICLES = 100;

const int RENDER_WIDTH = 40;
const int RENDER_HEIGHT = 20;
const int GRID_SIZE = 40;//static_cast<int>(SIMULATION_WIDTH / SMOOTHING_RADIUS);
const float GRAVITY_X_INITIAL = 9.81f;
const float GRAVITY_Y_INITIAL = 9.81f;
const float DAMPING = 0.98f;
const float FORCE_SCALE = 0.5f;

const int MAX_PARTICLES_PER_CELL = 10;  // Based on your simulation needs.
float GRAVITY_X = GRAVITY_X_INITIAL;
float GRAVITY_Y = GRAVITY_Y_INITIAL;


struct Vec2 {
    float x, y;

    Vec2() : x(0.0f), y(0.0f) {}
    Vec2(float x, float y) : x(x), y(y) {}

    Vec2 operator+(const Vec2 &other) const { return Vec2(x + other.x, y + other.y); }
    Vec2 operator-(const Vec2 &other) const { return Vec2(x - other.x, y - other.y); }
    Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
    Vec2 operator/(float scalar) const { return Vec2(x / scalar, y / scalar); }

    float length() const { return std::sqrt(x * x + y * y); }
    Vec2 normalized() const {
        float len = length();
        if(len == 0.0f) return Vec2(0.0f, 0.0f);
        return Vec2(x / len, y / len);
    }

    Vec2& operator+=(const Vec2 &other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    Vec2& operator*=(float scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
}

};

Vec2 operator*(float scalar, const Vec2& v) {
    return Vec2(v.x * scalar, v.y * scalar);
}

struct Particle {
    Vec2 position;
    Vec2 velocity;
    Vec2 force;
    float density;
    float pressure;
    float mass;
};

struct GridCell {
    int16_t particles[MAX_PARTICLES_PER_CELL];
    int count;
};

// std::vector<Particle> particles(NUM_PARTICLES);
Particle particles[NUM_PARTICLES];
GridCell grid[GRID_SIZE][GRID_SIZE];

float W(Vec2 r, float h) {
    float len = r.length();
    float q = len / h;
    if (q <= 1.0f)
        return (1.0f - q) * (1.0f - q);
    return 0.0f;
}

Vec2 gradientW(const Vec2& r, float h) {
    float len = r.length();
    float q = len / h;
    float factor;
    if (q <= 0.5f)
        factor = (-3.0f * q) + 2.5f;
    else
        factor = -0.5f / (q * q);

    return factor * r.normalized();
}

float laplacianW(float r, float h) {
    float q = r / h;
    if (q <= 0.5f)
        return (3.0f / (h * h)) * (1.0f - 3.0f * q);
    return (-1.0f / (q * h * h));
}

std::pair<int, int> getGridPosition(Vec2 position) {
    int x = static_cast<int>(position.x / SMOOTHING_RADIUS);
    int y = static_cast<int>(position.y / SMOOTHING_RADIUS);

    // Clamping the values to ensure they are within bounds
    x = std::max(0, std::min(x, GRID_SIZE - 1));
    y = std::max(0, std::min(y, GRID_SIZE - 1));

    return { x, y };
}

void updateGrid() {
    for (int i = 0; i < GRID_SIZE; i++)
        for (int j = 0; j < GRID_SIZE; j++)
            grid[i][j].count = 0;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        auto &p = particles[i];
        auto gridPos = getGridPosition(p.position);
        if (grid[gridPos.first][gridPos.second].count < MAX_PARTICLES_PER_CELL) {
            grid[gridPos.first][gridPos.second].particles[grid[gridPos.first][gridPos.second].count++] = i;
        }
        // Maybe handle count exceeding MAX_PARTICLES_PER_CELL?
    }
}

Particle** getNeighbors(Particle &p, int& neighborCount) {
    neighborCount = 0;
    auto gridPos = getGridPosition(p.position);
    // std::vector<Particle*> neighbors;
    static Particle* neighbors[NUM_PARTICLES];


    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int x = gridPos.first + i;
            int y = gridPos.second + j;
            if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
                for (int k = 0; k < grid[x][y].count; k++) {
                    // neighbors.push_back(&particles[grid[x][y].particles[k]]);
                    neighbors[neighborCount++] = &particles[grid[x][y].particles[k]];
                }
            }
        }
    }

    return neighbors;
}

void computeDensityPressure() {
    for (auto &pi : particles) {
        pi.density = 0.0f;

        int neighborCount;
        Particle** neighborArray = getNeighbors(pi, neighborCount);
        for (int n = 0; n < neighborCount; ++n) {
            Particle* pj = neighborArray[n];
            Vec2 r = pj->position - pi.position;
            pi.density += W(r, SMOOTHING_RADIUS);
        }

        pi.pressure = pi.density - 1.0f;  // Linear equation of state
    }
}

void computeForces() {
    float mu = 0.1f;

    for (Particle& pi : particles) {
        Vec2 f_pressure(0.0f, 0.0f);
        Vec2 f_viscosity(0.0f, 0.0f);
        Vec2 f_gravity(GRAVITY_X * pi.density, GRAVITY_Y * pi.density);

        int neighborCount;
        auto neighbors = getNeighbors(pi, neighborCount);
        for (int n = 0; n < neighborCount; ++n) {
            Particle* pj = neighbors[n];
            Vec2 r = pj->position - pi.position;
            float r_length = r.length();

            Vec2 v_diff = pj->velocity - pi.velocity;

            float pressure_scaler = -pj->mass * ((pi.pressure + pj->pressure) / (2 * pj->density));

            f_pressure += gradientW(r, SMOOTHING_RADIUS) * pressure_scaler;
            f_viscosity += v_diff * mu * pj->mass / pj->density * laplacianW(r_length, SMOOTHING_RADIUS);
        }

        pi.force = (f_pressure + f_viscosity + f_gravity) * FORCE_SCALE;
    }
}

void update() {
    updateGrid();
    computeDensityPressure();
    computeForces();

    const float deltaTime = 0.01f;

    for (auto &pi : particles) {
        pi.velocity.x += pi.force.x / pi.density * deltaTime;
        pi.velocity.y += pi.force.y / pi.density * deltaTime;

        pi.position.x += pi.velocity.x * deltaTime;
        pi.position.y += pi.velocity.y * deltaTime;
        pi.velocity *= DAMPING;


        // Boundary conditions
        if (pi.position.x < 0) {
            pi.position.x = 0;
            pi.velocity.x = -pi.velocity.x;  // Reflect the velocity
        }
        if (pi.position.y < 0) {
            pi.position.y = 0;
            pi.velocity.y = -pi.velocity.y;  // Reflect the velocity
        }
        if (pi.position.x > SIMULATION_WIDTH) {
            pi.position.x = SIMULATION_WIDTH;
            pi.velocity.x = -pi.velocity.x;  // Reflect the velocity
        }
        if (pi.position.y > SIMULATION_HEIGHT) {
            pi.position.y = SIMULATION_HEIGHT;
            pi.velocity.y = -pi.velocity.y;  // Reflect the velocity
        }
    }
}

void render() {
    char screen[RENDER_HEIGHT][RENDER_WIDTH];

    for (int i = 0; i < RENDER_HEIGHT; i++)
        for (int j = 0; j < RENDER_WIDTH; j++)
            screen[i][j] = ' ';

    for (auto &p : particles) {
        int x = static_cast<int>((p.position.x / SIMULATION_WIDTH) * RENDER_WIDTH);
        int y = static_cast<int>((p.position.y / SIMULATION_HEIGHT) * RENDER_HEIGHT);

        if (x >= 0 && x < RENDER_WIDTH && y >= 0 && y < RENDER_HEIGHT)
            screen[y][x] = '#';
    }

    for (int i = 0; i < RENDER_HEIGHT; i++) {
        for (int j = 0; j < RENDER_WIDTH; j++)
            std::cout << screen[i][j];
        std::cout << std::endl;
    }
    std::cout << "--------------------------------------------" << std::endl;  // Separator.
}

int main() {
    float theta = 0.0f;
    const float deltaTheta = M_PI / 180.0f * 0.2f; // 0.2 degree in radians
    // Initial setup
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].position = Vec2((static_cast<float>(rand()) / RAND_MAX) * (SIMULATION_WIDTH - 0.01f), (static_cast<float>(rand()) / RAND_MAX) * (SIMULATION_HEIGHT - 0.01f));
        particles[i].velocity = Vec2(0, 0);
        particles[i].mass = 1.0f;
    }

    while (true) {
        update();
        render();
        std::cout<<GRAVITY_X << " " << GRAVITY_Y << "\n" ;
        // std::cin.get();  // Wait for input before updating the screen
        usleep(10000);

        // change gravity
        GRAVITY_X = GRAVITY_X_INITIAL * cos(theta) - GRAVITY_Y_INITIAL * sin(theta);
        GRAVITY_Y = GRAVITY_X_INITIAL * sin(theta) + GRAVITY_Y_INITIAL * cos(theta);
        theta += deltaTheta;
        if (theta >= 2 * M_PI) {
            theta -= 2 * M_PI;  // Reset theta to keep it between 0 and 2*PI
        }

    }

    return 0;
}
