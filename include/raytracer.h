/*
 * raytracer.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#pragma once

////////////////////////////////////////////////////////////////////////////////
//
// Very simple ray tracing example
//
////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <ctime>
#ifdef __CUDACC__
#include <thrust/random.h>
#endif

#include "dmatrix.h"
#include "functors.h"
#include "argtype.h"
#include "muesli.h"

namespace msl {

namespace examples {

namespace raytracer {
#ifdef __CUDACC__
#define MAX(a, b)  fmaxf(a, b)
#define MIN(a, b)  fminf(a, b)
#define POW(a, b)  powf(a, b)
#define SQRT(a)    sqrtf(a)
#define FMOD(a, b) fmodf(a, b)
#define COS(a)     cosf(a)
#define SIN(a)     sinf(a)
#define TAN(a)     tanf(a)
#else
#define MAX(a, b)  std::max(a, b)
#define MIN(a, b)  std::min(a, b)
#define POW(a, b)  std::pow(a, b)
#define SQRT(a)    std::sqrt(a)
#define FMOD(a, b) std::fmod(a, b)
#define COS(a)     std::cos(a)
#define SIN(a)     std::sin(a)
#define TAN(a)     std::tan(a)
#endif
const float kPi = 3.1415926535f;

//
// RGB color class
//

struct Color
{
  float m_r, m_g, m_b;

  MSL_USERFUNC
  Color()
    : m_r(0.0f), m_g(0.0f), m_b(0.0f)
  {
  }

  MSL_USERFUNC
  Color(const Color& c)
    : m_r(c.m_r), m_g(c.m_g), m_b(c.m_b)
  {
  }

  MSL_USERFUNC
  Color(float r, float g, float b)
    : m_r(r), m_g(g), m_b(b)
  {
  }

  MSL_USERFUNC
  explicit Color(float f)
    : m_r(f), m_g(f), m_b(f)
  {
  }

  MSL_USERFUNC
  void clamp(float p_min = 0.0f, float p_max = 1.0f)
  {
    m_r = MAX(p_min, MIN(p_max, m_r));
    m_g = MAX(p_min, MIN(p_max, m_g));
    m_b = MAX(p_min, MIN(p_max, m_b));
  }

  MSL_USERFUNC
  Color& operator =(const Color& c)
  {
    m_r = c.m_r;
    m_g = c.m_g;
    m_b = c.m_b;
    return *this;
  }

  MSL_USERFUNC
  Color& operator +=(const Color& c)
  {
    m_r += c.m_r;
    m_g += c.m_g;
    m_b += c.m_b;
    return *this;
  }

  MSL_USERFUNC
  Color& operator -=(const Color& c)
  {
    m_r -= c.m_r;
    m_g -= c.m_g;
    m_b -= c.m_b;
    return *this;
  }

  MSL_USERFUNC
  Color& operator *=(const Color& c)
  {
    m_r *= c.m_r;
    m_g *= c.m_g;
    m_b *= c.m_b;
    return *this;
  }

  MSL_USERFUNC
  Color& operator /=(const Color& c)
  {
    m_r /= c.m_r;
    m_g /= c.m_g;
    m_b /= c.m_b;
    return *this;
  }

  MSL_USERFUNC
  Color& operator *=(float f)
  {
    m_r *= f;
    m_g *= f;
    m_b *= f;
    return *this;
  }

  MSL_USERFUNC
  Color& operator /=(float f)
  {
    m_r /= f;
    m_g /= f;
    m_b /= f;
    return *this;
  }
};

MSL_USERFUNC
Color operator +(const Color& c1, const Color& c2)
{
  return Color(c1.m_r + c2.m_r, c1.m_g + c2.m_g, c1.m_b + c2.m_b);
}

MSL_USERFUNC
Color operator -(const Color& c1, const Color& c2)
{
  return Color(c1.m_r - c2.m_r, c1.m_g - c2.m_g, c1.m_b - c2.m_b);
}

MSL_USERFUNC
Color operator *(const Color& c1, const Color& c2)
{
  return Color(c1.m_r * c2.m_r, c1.m_g * c2.m_g, c1.m_b * c2.m_b);
}

MSL_USERFUNC
Color operator /(const Color& c1, const Color& c2)
{
  return Color(c1.m_r / c2.m_r, c1.m_g / c2.m_g, c1.m_b / c2.m_b);
}

MSL_USERFUNC
Color operator *(const Color& c, float f)
{
  return Color(f * c.m_r, f * c.m_g, f * c.m_b);
}

MSL_USERFUNC
Color operator *(float f, const Color& c)
{
  return Color(f * c.m_r, f * c.m_g, f * c.m_b);
}

MSL_USERFUNC
Color operator /(const Color& c, float f)
{
  return Color(c.m_r / f, c.m_g / f, c.m_b / f);
}

struct Pixel
{
  float m_r;
  float m_g;
  float m_b;
  size_t xpos;
  size_t ypos;

  MSL_USERFUNC
  Pixel()
    : m_r(0.0f), m_g(0.0f), m_b(0.0f), xpos(0), ypos(0)
  {
  }

  MSL_USERFUNC
  Pixel(float p_r, float p_g, float p_b, size_t x, size_t y)
    : m_r(p_r), m_g(p_g), m_b(p_b), xpos(x), ypos(y)
  {
  }

  MSL_USERFUNC
  void setColor(Color& c)
  {
    m_r = c.m_r;
    m_g = c.m_g;
    m_b = c.m_b;
  }
};

//
// 3D vector class
//

struct Vector
{
  float m_x, m_y, m_z;

  MSL_USERFUNC
  Vector()
    : m_x(0.0f), m_y(0.0f), m_z(0.0f)
  {
  }

  MSL_USERFUNC
  Vector(const Vector& v)
    : m_x(v.m_x), m_y(v.m_y), m_z(v.m_z)
  {
  }

  MSL_USERFUNC
  Vector(float x, float y, float z)
    : m_x(x), m_y(y), m_z(z)
  {
  }

  MSL_USERFUNC
  explicit Vector(float f)
          : m_x(f), m_y(f), m_z(f)
  {
  }

  MSL_USERFUNC
  float length2() const
  {
    return m_x * m_x + m_y * m_y + m_z * m_z;
  }

  MSL_USERFUNC
  float length() const
  {
    return SQRT(length2());
  }

  // Returns old length from before normalization
  MSL_USERFUNC
  float normalize()
  {
    float len = length();
    *this /= len;
    return len;
  }

  // Return a vector in this same direction, but normalized
  MSL_USERFUNC
  Vector normalized() const
  {
    Vector r(*this);
    r.normalize();
    return r;
  }

  MSL_USERFUNC
  Vector& operator =(const Vector& v)
  {
    m_x = v.m_x;
    m_y = v.m_y;
    m_z = v.m_z;
    return *this;
  }

  MSL_USERFUNC
  Vector& operator +=(const Vector& v)
  {
    m_x += v.m_x;
    m_y += v.m_y;
    m_z += v.m_z;
    return *this;
  }

  MSL_USERFUNC
  Vector& operator -=(const Vector& v)
  {
    m_x -= v.m_x;
    m_y -= v.m_y;
    m_z -= v.m_z;
    return *this;
  }

  MSL_USERFUNC
  Vector& operator *=(float f)
  {
    m_x *= f;
    m_y *= f;
    m_z *= f;
    return *this;
  }

  MSL_USERFUNC
  Vector& operator /=(float f)
  {
    m_x /= f;
    m_y /= f;
    m_z /= f;
    return *this;
  }
};

MSL_USERFUNC
Vector operator +(const Vector& v1, const Vector& v2)
{
  return Vector(v1.m_x + v2.m_x, v1.m_y + v2.m_y, v1.m_z + v2.m_z);
}

MSL_USERFUNC
Vector operator -(const Vector& v1, const Vector& v2)
{
  return Vector(v1.m_x - v2.m_x, v1.m_y - v2.m_y, v1.m_z - v2.m_z);
}

MSL_USERFUNC
Vector operator *(const Vector& v, float f)
{
  return Vector(f * v.m_x, f * v.m_y, f * v.m_z);
}

MSL_USERFUNC
Vector operator *(float f, const Vector& v)
{
  return Vector(f * v.m_x, f * v.m_y, f * v.m_z);
}

// dot(v1, v2) = length(v1) * length(v2) * cos(angle between v1, v2)
MSL_USERFUNC
float dot(const Vector& v1, const Vector& v2)
{
  return v1.m_x * v2.m_x + v1.m_y * v2.m_y + v1.m_z * v2.m_z;
}

// cross(v1, v2) = length(v1) * length(v2) * sin(angle between v1, v2);
MSL_USERFUNC
Vector cross(const Vector& v1, const Vector& v2)
{
  return Vector(v1.m_y * v2.m_z - v1.m_z * v2.m_y, v1.m_z * v2.m_x - v1.m_x * v2.m_z, v1.m_x * v2.m_y - v1.m_y * v2.m_x);
}

// A few debug print helpers for Color and Vector, in case we need them.
std::ostream&
operator <<(std::ostream& stream, const Color& c)
{
  stream << '(' << c.m_r << ", " << c.m_g << ", " << c.m_b << ')';
  return stream;
}

std::ostream&
operator <<(std::ostream& stream, const Vector& v)
{
  stream << '[' << v.m_x << ", " << v.m_y << ", " << v.m_z << ']';
  return stream;
}

typedef Vector Point;

//
// Ray
//

const float kRayTMin = 0.00001f;
const float kRayTMax = 1.0e30f;

struct Ray
{
  Point m_origin;
  Vector m_direction;
  float m_tMax;

  MSL_USERFUNC
  Ray()
    : m_origin(), m_direction(0.0f, 0.0f, 1.0f), m_tMax(kRayTMax)
  {
  }

  MSL_USERFUNC
  Ray(const Ray& r)
    : m_origin(r.m_origin), m_direction(r.m_direction), m_tMax(r.m_tMax)
  {
  }

  MSL_USERFUNC
  Ray(const Point& origin, const Vector& direction, float tMax = kRayTMax)
    : m_origin(origin), m_direction(direction), m_tMax(tMax)
  {
  }

  MSL_USERFUNC
  Ray& operator =(const Ray& r)
  {
    m_origin = r.m_origin;
    m_direction = r.m_direction;
    m_tMax = r.m_tMax;
    return *this;
  }

  MSL_USERFUNC
  Point calculate(float t) const
  {
    return m_origin + t * m_direction;
  }
};

//
// Intersection (results from casting a ray)
//

// forward declarations needed to declare pointers to.
struct Sphere;
struct RectangleLight;
struct Lambert;
struct Plane;
struct Emitter;
struct Phong;

struct Intersection
{
  Ray m_ray;
  float m_t;
  RectangleLight* m_rectLight;
  Sphere* m_sphere;
  Lambert* m_Lambert;
  Plane* m_Plane;
  Phong* m_Phong;
  Emitter* m_Emitter;
  Color m_colorModifier;
  Vector m_normal;

  MSL_USERFUNC
  Intersection()
    : m_ray(), m_t(kRayTMax), m_rectLight(0), m_sphere(0), m_Lambert(0), m_Plane(0), m_Phong(0), m_Emitter(0),
      m_colorModifier(1.0f, 1.0f, 1.0f), m_normal()
  {
  }

  MSL_USERFUNC
  Intersection(const Intersection& i)
    : m_ray(i.m_ray), m_t(i.m_t), m_rectLight(i.m_rectLight), m_sphere(i.m_sphere), m_Lambert(i.m_Lambert),
      m_Plane(i.m_Plane), m_Phong(i.m_Phong), m_Emitter(0), m_colorModifier(i.m_colorModifier), m_normal(i.m_normal)
  {
  }

  MSL_USERFUNC
  Intersection(const Ray& ray)
    : m_ray(ray), m_t(ray.m_tMax), m_rectLight(0), m_sphere(0), m_Lambert(0), m_Plane(0), m_Phong(NULL),
      m_Emitter(0), m_colorModifier(1.0f, 1.0f, 1.0f), m_normal()
  {
  }

  MSL_USERFUNC
  Intersection& operator =(const Intersection& i)
  {
    m_ray = i.m_ray;
    m_t = i.m_t;
    m_rectLight = i.m_rectLight;
    m_sphere = i.m_sphere;
    m_Lambert = i.m_Lambert;
    m_Plane = i.m_Plane;
    m_Phong = i.m_Phong;
    m_colorModifier = i.m_colorModifier;
    m_normal = i.m_normal;
    return *this;
  }

  //bool intersected() const { return (m_pShape == NULL) ? false : true; }

  MSL_USERFUNC
  Point position() const
  {
    return m_ray.calculate(m_t);
  }
};

//
// Material
//

// Lambertian diffuse material
struct Lambert
{
  Color m_color;

  MSL_USERFUNC
  Lambert(const Color& color)
    : m_color(color)
  {
  }

  MSL_USERFUNC
  Lambert()
  {
    m_color = Color();
  }

  MSL_USERFUNC
  Color shade(const Point& position, const Vector& normal, const Vector& incomingRayDirection,
          const Vector& lightDirectionNorm)
  {
    float dot_tmp = dot(lightDirectionNorm, normal);
    return MAX(0.0f, dot_tmp) * m_color;
  }

  MSL_USERFUNC
  Color emittance()
  {
    return Color();
  }
};

// Phong glossy material
struct Phong
{
  Color m_color;
  float m_exponent;

  MSL_USERFUNC
  Phong(const Color& color, float exponent)
    : m_color(color), m_exponent(exponent)
  {
  }

  MSL_USERFUNC
  Phong()
  {
    m_color = Color();
    m_exponent = 1.0f;
  }

  MSL_USERFUNC
  Color shade(const Point& position, const Vector& normal, const Vector& incomingRayDirection,
          const Vector& lightDirectionNorm)
  {
    float dot_tmp = dot(lightDirectionNorm, normal);
    return POW(MAX(0.0f, dot_tmp), m_exponent) * m_color;
  }

  MSL_USERFUNC
  Color emittance()
  {
    return Color();
  }
};

// Emitter (light) material
struct Emitter
{
  Color m_color;
  float m_power;

  MSL_USERFUNC
  Emitter(const Color& color, float power)
    : m_color(color), m_power(power)
  {
  }

  MSL_USERFUNC
  Emitter()
    : m_power(0.0f)
  {
  }

  MSL_USERFUNC
  Color emittance()
  {
    return m_color * m_power;
  }

  MSL_USERFUNC
  Color shade(const Point& position, const Vector& normal, const Vector& incomingRayDirection,
          const Vector& lightDirectionNorm)
  {
    return Color();
  }
};

//
// Shapes (scene hierarchy)
//

// Sphere
struct Sphere
{
  Point m_position;
  float m_radius;
  Lambert m_Lambert;
  Phong m_Phong;
  bool lambert;

  MSL_USERFUNC
  Sphere(const Point& position, float radius, Lambert& pLambert)
    : m_position(position), m_radius(radius), m_Lambert(pLambert), m_Phong(Color(), 1.0f), lambert(1)
  {
  }

  MSL_USERFUNC
  Sphere(const Point& position, float radius, Phong& pPhong)
    : m_position(position), m_radius(radius), m_Lambert(Color()), m_Phong(pPhong), lambert(0)
  {
  }

  MSL_USERFUNC
  Sphere()
    : m_radius(0.0f), m_Lambert(Lambert()), lambert(0)
  {
  }

  MSL_USERFUNC
  bool intersect(Intersection& intersection)
  {
    Ray localRay = intersection.m_ray;
    localRay.m_origin -= m_position;

    // Ray-sphere intersection can result in either zero, one or two points
    // of intersection. It turns into a quadratic equation, so we just find
    // the solution using the quadratic formula.  Note that there is a
    // slightly more stable form of it when computing it on a computer, and
    // we use that method to keep everything accurate.

    // Calculate quadratic coeffs
    float a = localRay.m_direction.length2();
    float b = 2.0f * dot(localRay.m_direction, localRay.m_origin);
    float c = localRay.m_origin.length2() - m_radius * m_radius;

    float t0, t1, discriminant;
    discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
      // Discriminant less than zero?  No solution => no intersection.
      return false;
    }
    discriminant = SQRT(discriminant);

    // Compute a more stable form of our param t (t0 = q/a, t1 = c/q)
    // q = -0.5 * (b - sqrt(b * b - 4.0 * a * c)) if b < 0, or
    // q = -0.5 * (b + sqrt(b * b - 4.0 * a * c)) if b >= 0
    float q;
    if (b < 0.0f) {
      q = -0.5f * (b - discriminant);
    } else {
      q = -0.5f * (b + discriminant);
    }

    // Get our final parametric values
    t0 = q / a;
    if (q != 0.0f) {
      t1 = c / q;
    } else {
      t1 = intersection.m_t;
    }

    // Swap them so they are ordered right
    if (t0 > t1) {
      float temp = t1;
      t1 = t0;
      t0 = temp;
    }

    // Check our intersection for validity against this ray's extents
    if (t0 >= intersection.m_t || t1 < kRayTMin) {
      return false;
    }

    if (t0 >= kRayTMin) {
      intersection.m_t = t0;
    } else if (t1 < intersection.m_t) {
      intersection.m_t = t1;
    } else {
      return false;
    }

    // Create our intersection data
    Point localPos = localRay.calculate(intersection.m_t);
    Vector worldNorm = localPos.normalized();

    intersection.m_sphere = this;
    if (lambert) {
      intersection.m_Lambert = &m_Lambert;
      intersection.m_Phong = 0;
    } else {
      intersection.m_Phong = &m_Phong;
      intersection.m_Lambert = 0;
    }
    intersection.m_normal = worldNorm;
    intersection.m_colorModifier = Color(1.0f, 1.0f, 1.0f);

    return true;
  }

  // Given two random numbers between 0.0 and 1.0, find a location + surface
  // normal on the surface of the *light*.
  MSL_USERFUNC
  bool sampleSurface(float u1, float u2, const Point& referencePosition, Point& outPosition, Vector& outNormal)
  {
    // Pick a random point on the whole sphere surface
    outNormal = uniformToSphere(u1, u2);
    outPosition = outNormal * m_radius + m_position;
    if (dot(outNormal, referencePosition - outPosition) < 0.0f) {
      // Point was on the opposite side?  Flip around to make this
      // more reasonable.
      outNormal *= -1.0f;
      outPosition = outNormal * m_radius + m_position;
    }
    return true;
  }

  // Helper method for finding a random point on the sphere
  MSL_USERFUNC
  Vector uniformToSphere(float u1, float u2)
  {
    // Find a height uniformly distributed on the sphere
    float z = 1.0f - 2.0f * u1;
    // Find the radius based on that height that sits on the sphere surface
    float radius = SQRT(MAX(0.0f, 1.0f - z * z));
    // Find a random angle around the sphere's equator
    float phi = kPi * 2.0f * u2;
    // And put it all together...
    return Vector(radius * COS(phi), radius * SIN(phi), z);
  }
};

// Area light with a corner and two sides to define a rectangular/parallelipiped shape
struct RectangleLight
{
  Point m_position;
  Vector m_side1, m_side2;
  Emitter m_emitter;
  Color m_color;
  float m_power;

  MSL_USERFUNC
  RectangleLight(const Point& pos, const Vector& side1, const Vector& side2, const Color& color,
          float power)
    : m_position(pos), m_side1(side1), m_side2(side2), m_emitter(color, power), m_color(color), m_power(power)
  {
  }

  MSL_USERFUNC
  RectangleLight()
    : m_power(0.0f)
  {
  }

  MSL_USERFUNC
  bool intersect(Intersection& intersection)
  {
    // This is much like a plane intersection, except we also range check it
    // to make sure it's within the rectangle.  Please see the plane shape
    // intersection method for a little more info.

    Vector normal = cross(m_side1, m_side2).normalized();
    float nDotD = dot(normal, intersection.m_ray.m_direction);
    if (nDotD == 0.0f) {
      return false;
    }

    float t = (dot(m_position, normal) - dot(intersection.m_ray.m_origin, normal))
            / dot(intersection.m_ray.m_direction, normal);

    // Make sure t is not behind the ray, and is closer than the current
    // closest intersection.
    if (t >= intersection.m_t || t < kRayTMin) {
      return false;
    }

    // Take the intersection point on the plane and transform it to a local
    // space where we can really easily check if it's in range or not.
    Vector side1Norm = m_side1;
    Vector side2Norm = m_side2;
    float side1Length = side1Norm.normalize();
    float side2Length = side2Norm.normalize();

    Point worldPoint = intersection.m_ray.calculate(t);
    Point worldRelativePoint = worldPoint - m_position;
    Point localPoint = Point(dot(worldRelativePoint, side1Norm), dot(worldRelativePoint, side2Norm), 0.0f);

    // Do the actual range check
    if (localPoint.m_x < 0.0f || localPoint.m_x > side1Length || localPoint.m_y < 0.0f
            || localPoint.m_y > side2Length) {
      return false;
    }

    // This intersection is the closest so far, so record it.
    intersection.m_t = t;
    intersection.m_rectLight = this;
    intersection.m_Emitter = &m_emitter;
    intersection.m_colorModifier = Color(1.0f, 1.0f, 1.0f);
    intersection.m_normal = normal;
    // Hit the back side of the light?  We'll count it, so flip the normal
    // to effectively make a double-sided light.
    if (dot(intersection.m_normal, intersection.m_ray.m_direction) > 0.0f) {
      intersection.m_normal *= -1.0f;
    }

    return true;
  }

  // Given two random numbers between 0.0 and 1.0, find a location + surface
  // normal on the surface of the *light*.
  MSL_USERFUNC
  bool sampleSurface(float u1, float u2, const Point& referencePosition, Point& outPosition, Vector& outNormal)
  {
    outNormal = cross(m_side1, m_side2).normalized();
    outPosition = m_position + m_side1 * u1 + m_side2 * u2;
    // Reference point out in back of the light?  That's okay, we'll flip
    // the normal to have a double-sided light.
    if (dot(outNormal, outPosition - referencePosition) > 0.0f) {
      outNormal *= -1.0f;
    }
    return true;
  }

  MSL_USERFUNC
  Color emitted() const
  {
    return m_color * m_power;
  }
};

// Infinite-extent plane, with option bullseye texturing to make it interesting.
struct Plane
{
  Point m_position;
  Vector m_normal;
  Lambert m_Lambert;
  bool m_bullseye;

  MSL_USERFUNC
  Plane(const Point& position, const Vector& normal, Lambert& pLambert, bool bullseye = false)
    : m_position(position), m_normal(normal.normalized()), m_Lambert(pLambert), m_bullseye(bullseye)
  {
  }

  MSL_USERFUNC
  Plane()
    : m_bullseye(0)
  {
  }

  MSL_USERFUNC
  bool intersect(Intersection& intersection)
  {
    // Plane eqn: ax+by+cz+d=0; another way of writing it is: dot(n, p-p0)=0
    // where n=normal=(a,b,c), and p=(x,y,z), and p0 is position.  Now, p is
    // the ray equation (the intersection point is along the ray): p=origin+t*direction
    // So the plane-ray intersection eqn is dot(n, origin+t*direction-p0)=0.
    // Distribute, and you get:
    //     dot(n, origin) + t*dot(n, direction) - dot(n, p0) = 0
    // Solve for t, and you get:
    //    t = (dot(n, p0) - dot(n, origin)) / dot(n, direction)

    // Check if it's even possible to intersect
    float nDotD = dot(m_normal, intersection.m_ray.m_direction);
    if (nDotD >= 0.0f) {
      return false;
    }

    float t = (dot(m_position, m_normal) - dot(intersection.m_ray.m_origin, m_normal))
            / dot(intersection.m_ray.m_direction, m_normal);

    // Make sure t is not behind the ray, and is closer than the current
    // closest intersection.
    if (t >= intersection.m_t || t < kRayTMin) {
      return false;
    }

    // This intersection is closer, so record it.
    intersection.m_t = t;
    intersection.m_Plane = this;
    intersection.m_Lambert = &m_Lambert;
    intersection.m_Phong = 0;
    intersection.m_normal = m_normal;
    intersection.m_colorModifier = Color(1.0f, 1.0f, 1.0f);

    // Hack bullseye pattern to get some variation
    if (m_bullseye && FMOD((intersection.position() - m_position).length() * 0.25f,
            1.0f) > 0.5f) {
      intersection.m_colorModifier *= 0.2f;
    }

    return true;
  }
};

// List of shapes, so you can aggregate a pile of them
struct ShapeSet
{
  size_t nLights;
  size_t nSpheres;
  size_t l_current;
  size_t s_current;
  RectangleLight* lights;
  Sphere* spheres;
  Plane plane;

  ShapeSet(size_t p_lights, size_t p_spheres)
          : nLights(p_lights), nSpheres(p_spheres), l_current(0), s_current(0)
  {
    lights = new RectangleLight[nLights];
    spheres = new Sphere[nSpheres];
  }

  ShapeSet()
          : nLights(0), nSpheres(0), l_current(0), s_current(0), lights(0), spheres(0)
  {
  }

  ~ShapeSet()
  {
    delete[] spheres;
    delete[] lights;
  }

  MSL_USERFUNC
  bool intersect(Intersection& intersection)
  {
    bool intersectedAny = false;
    for (size_t i = 0; i < nSpheres; i++) {
      if (spheres[i].intersect(intersection)) {
        intersectedAny = 1;
      }
    }
    for (size_t i = 0; i < nLights; i++) {
      if (lights[i].intersect(intersection)) {
        intersectedAny = 1;
      }
    }
    if (plane.intersect(intersection)) {
      intersectedAny = 1;
    }
    return intersectedAny;
  }

  void addLight(RectangleLight& pLight)
  {
    lights[l_current++] = pLight;
  }

  void addSphere(Sphere& pSphere)
  {
    spheres[s_current++] = pSphere;
  }

  void addPlane(Plane& pPlane)
  {
    plane = pPlane;
  }

  void clearShapes()
  {
    delete[] spheres;
    delete[] lights;
    spheres = 0;
    lights = 0;
  }
};

// Set up a camera ray given the look-at spec, FOV, and screen position to aim at.
MSL_USERFUNC
Ray makeCameraRay(float fieldOfViewInDegrees, const Point& origin, const Vector& target,
        const Vector& targetUpDirection, float xScreenPos0To1, float yScreenPos0To1)
{
  Vector forward = (target - origin).normalized();
  Vector right = cross(forward, targetUpDirection).normalized();
  Vector up = cross(right, forward).normalized();

  // Convert to radians, as that is what the math calls expect
  float tanFov = std::tan(fieldOfViewInDegrees * kPi / 180.0f);

  Ray ray;

  // Set up ray info
  ray.m_origin = origin;
  ray.m_direction = forward + right * ((xScreenPos0To1 - 0.5f) * tanFov) + up * ((yScreenPos0To1 - 0.5f) * tanFov);
  ray.m_direction.normalize();

  return ray;
}

// Available materials
//Lambert blueishLambert(Color(0.1f, 0.1f, 1.0f));
//Lambert purplishLambert(Color(0.9f, 0.7f, 0.8f));
//Lambert ml(Color(0.0f, 0.4f, 0.0f));
//Phong greenishPhong(Color(0.7f, 0.9f, 0.7f), 16.0f);
//Phong mp(Color(0.8f, 0.5f, 0.1f), 16.0f);

Lambert blueishLambert(Color(0.9f, 0.9f, 1.0f));
Lambert purplishLambert(Color(0.9f, 0.7f, 0.8f));
Lambert ml(Color(0.0f, 0.4f, 0.0f));
Phong greenishPhong(Color(0.7f, 0.9f, 0.7f), 16.0f);
Phong mp(Color(0.8f, 0.5f, 0.1f), 16.0f);
Emitter me(Color(0.2f, 0.5f, 1.0f), 1.0f);

void createScene(ShapeSet* scene, size_t nSpheres)
{
  Plane plane(Point(0.0f, -2.0f, 0.0f), Vector(0.0f, 1.0f, 0.0f), blueishLambert, 1);
  scene->addPlane(plane);

//  Sphere sphere(Point(-5.0f, 0.0f, 0.0f), 0.95f, purplishLambert);
//  scene->addSphere(sphere);
//
//  Sphere sphere1(Point(0.0f, 0.0f, 5.0f), 0.95f, purplishLambert);
//  scene->addSphere(sphere1);
//
//  Sphere sphere2(Point(5.0f, 0.0f, 0.0f), 0.95f, blueishLambert);
//  scene->addSphere(sphere2);
//
//  Sphere sphere3(Point(0.0f, 0.0f, -5.0f), 0.95f, greenishPhong);
//  scene->addSphere(sphere3);
//
//  Sphere sphere4(Point(0.0f, 0.0f, 0.0f), 0.95f, greenishPhong);
//  scene->addSphere(sphere4);

  float dia;
  for (size_t i = 0; i < nSpheres; i++) {
    float x = float(rand() % 10) * rand() / (float) RAND_MAX - float(rand() % 10);
    float y = float(rand() % 10) * rand() / (float) RAND_MAX;
    float z = float(rand() % 10) * rand() / (float) RAND_MAX - float(rand() % 10);
    dia = rand() / (float) RAND_MAX;
    Sphere sphere;
    if ((rand() % 10) < 5) {
      sphere = Sphere(Point(x, y, z), dia, blueishLambert);
    } else {
      sphere = Sphere(Point(x, y, z), dia, greenishPhong);
    }
    scene->addSphere(sphere);
  }

  RectangleLight areaLight(Point(-3.0f, 1000.0f, 9.0f), Vector(5.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 5.0f),
          Color(1.0f, 1.0f, 1.0f), 1.02f);
  RectangleLight areaLight1(Point(-6.0f, 1000.0f, 9.0f), Vector(5.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 5.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);
  RectangleLight areaLight2(Point(0.0f, 1000.0f, 9.0f), Vector(5.0f, 0.0f, 0.0f), Vector(0.0f, 0.0f, 5.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);
  RectangleLight areaLight3(Point(4.0f, 2000.0f, 15.0f), Vector(0.0f, 5.0f, 0.0f), Vector(2.0f, 0.0f, -2.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);

  RectangleLight areaLight4(Point(4.0f, 2000.0f, -15.0f), Vector(0.0f, 5.0f, 0.0f), Vector(2.0f, 0.0f, -2.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);

  RectangleLight areaLight5(Point(8.0f, 2000.0f, 12.0f), Vector(0.0f, 5.0f, 0.0f), Vector(2.0f, 0.0f, -2.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);

  RectangleLight areaLight6(Point(-8.0f, 2000.0f, 12.0f), Vector(0.0f, 5.0f, 0.0f), Vector(2.0f, 0.0f, -2.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);

  RectangleLight areaLight7(Point(4.0f, 2000.0f, -12.0f), Vector(0.0f, 5.0f, 0.0f), Vector(2.0f, 0.0f, -2.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);

  RectangleLight areaLight8(Point(-1.0f, 2000.0f, -1.0f), Vector(0.0f, 5.0f, 0.0f), Vector(2.0f, 0.0f, -2.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);

  RectangleLight areaLight9(Point(0.0f, 2000.0f, 0.0f), Vector(0.0f, 5.0f, 0.0f), Vector(2.0f, 0.0f, -2.0f),
          Color(1.0f, 1.0f, 1.0f), 1.0f);

  scene->addLight(areaLight);
  scene->addLight(areaLight1);
  scene->addLight(areaLight2);
  scene->addLight(areaLight3);
  scene->addLight(areaLight4);
  scene->addLight(areaLight5);
  scene->addLight(areaLight6);
  scene->addLight(areaLight7);
  scene->addLight(areaLight8);
  scene->addLight(areaLight9);
}

class Scene
{
public:
  Scene(int nSpheres, int nLights)
          : shapeset(nLights, nSpheres), uploaded(0)
  {
    createScene(&shapeset, nSpheres);
    shapesets = new ShapeSet*[msl::Muesli::num_gpus];
  }

  ~Scene()
  {
#ifdef __CUDACC__
    if (uploaded) {
      for (int i = 0; i < Muesli::num_gpus; i++) {
        CUDA_CHECK_RETURN(cudaFree(shapesets[i]));
      }
    }
#endif
    delete[] shapesets;
  }

  std::vector<ShapeSet*> upload()
  {
    std::vector<ShapeSet*> ssv;

#ifdef __CUDACC__
    size_t sLights, sSpheres;
    for (int i = 0; i < msl::Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      CUDA_CHECK_RETURN(cudaMalloc((void ** ) &shapesets[i], sizeof(ShapeSet)));
      CUDA_CHECK_RETURN(cudaMemcpy(shapesets[i], &shapeset, sizeof(ShapeSet), cudaMemcpyHostToDevice));

      RectangleLight* d_lights;
      sLights = sizeof(RectangleLight) * shapeset.nLights;
      CUDA_CHECK_RETURN(cudaMalloc((void ** ) &d_lights, sLights));
      CUDA_CHECK_RETURN(cudaMemcpy(d_lights, shapeset.lights, sLights, cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(&(shapesets[i]->lights), &d_lights, sizeof(RectangleLight*), cudaMemcpyHostToDevice));

      Sphere* d_spheres;
      sSpheres = sizeof(Sphere) * shapeset.nSpheres;
      CUDA_CHECK_RETURN(cudaMalloc((void ** ) &d_spheres, sSpheres));
      CUDA_CHECK_RETURN(cudaMemcpy(d_spheres, shapeset.spheres, sSpheres, cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(&(shapesets[i]->spheres), &d_spheres, sizeof(Sphere*), cudaMemcpyHostToDevice));

      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      ssv.push_back(shapesets[i]);
    }
#else
    ssv.push_back(&shapeset);
#endif

    uploaded = 1;

    return ssv;
  }

  ShapeSet* getShapeSet()
  {
    return &shapeset;
  }

private:
  ShapeSet shapeset;
  ShapeSet** shapesets;
  bool uploaded;
};

class LScene: public msl::ArgumentType
{
public:
  LScene(Scene& scene)
    : shapeset_gpu(0), s(&scene), current_device(-1)
  {
    shapeset_cpu = scene.getShapeSet();
  }

  virtual ~LScene()
  {
  }

  virtual void update()
  {
    if (current_device == -1) {
      shapesets = s->upload();
    }
    current_device = (current_device + 1) % Muesli::num_gpus;
    shapeset_gpu = shapesets[current_device];
  }

  MSL_USERFUNC
  ShapeSet* getShapeSet() const
  {
#ifdef __CUDA_ARCH__
    return shapeset_gpu;
#else
    return shapeset_cpu;
#endif
  }

private:
  ShapeSet* shapeset_gpu, *shapeset_cpu;
  std::vector<ShapeSet*> shapesets;
  Scene* s;
  mutable int current_device;
};

void printRGB(msl::DMatrix<Pixel>& rtImg)
{
  size_t width = rtImg.getCols();
  size_t height = rtImg.getRows();

  // Set up the output file
  std::ostringstream headerStream;
  headerStream << "P6\n";
  headerStream << height << ' ' << width << '\n';
  headerStream << "255\n";
  std::ofstream fileStream("scene.ppm", std::ios::out | std::ios::binary);
  fileStream << headerStream.str();

  // Gather the rendered image
  Pixel** img = new Pixel*[height];
  for (size_t i = 0; i < height; i++) {
    img[i] = new Pixel[width];
  }
  rtImg.gather(img);

  Color pixelColor;
  for (size_t y = 0; y < width; y++) {
    for (size_t x = 0; x < height; ++x) {
      pixelColor.m_r = img[x][y].m_r;
      pixelColor.m_g = img[x][y].m_g;
      pixelColor.m_b = img[x][y].m_b;
      pixelColor.clamp();

      // Get 24-bit pixel sample and write it out
      unsigned char r, g, b;
      r = static_cast<unsigned char>(pixelColor.m_r * 255.0f);
      g = static_cast<unsigned char>(pixelColor.m_g * 255.0f);
      b = static_cast<unsigned char>(pixelColor.m_b * 255.0f);
      fileStream << r << g << b;
    }
  }

  for (size_t i = 0; i < height; i++) {
    delete[] img[i];
  }
  delete[] img;

  // Tidy up (probably unnecessary)
  fileStream.flush();
  fileStream.close();
}

void printGrayscale(msl::DMatrix<Pixel>& rtImg)
{
  size_t width = rtImg.getCols();
  size_t height = rtImg.getRows();

  // Set up the output file
  std::ostringstream headerStream;
  headerStream << "P2\n";
  headerStream << height << ' ' << width << '\n';
  headerStream << "255\n";
  std::ofstream fileStream("scene.pgm", std::ios::out | std::ios::binary);
  fileStream << headerStream.str();

  // Gather the rendered image
  Pixel** img = new Pixel*[height];
  for (size_t i = 0; i < height; i++) {
    img[i] = new Pixel[width];
  }
  rtImg.gather(img);

  Color pixelColor;
  for (size_t y = 0; y < width; y++) {
    for (size_t x = 0; x < height; ++x) {
      pixelColor.m_r = img[x][y].m_r;
      pixelColor.m_g = img[x][y].m_g;
      pixelColor.m_b = img[x][y].m_b;
      pixelColor.clamp();

      int gs = (int) ((pixelColor.m_r * 0.21f + pixelColor.m_g * 0.72f + pixelColor.m_b * 0.07f) * 255.0f);
      fileStream << gs << std::endl;
    }
  }

  for (size_t i = 0; i < height; i++) {
    delete[] img[i];
  }
  delete[] img;

  // Tidy up (probably unnecessary)
  fileStream.flush();
  fileStream.close();
}

} // namespace raytracer
} // namespace examples
} // namespace msl
