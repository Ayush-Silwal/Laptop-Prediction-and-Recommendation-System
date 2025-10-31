-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Oct 31, 2025 at 11:50 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `laptop_database`
--

-- --------------------------------------------------------

--
-- Table structure for table `bookings`
--

CREATE TABLE `bookings` (
  `bid` int(11) NOT NULL,
  `uid` int(11) NOT NULL,
  `rid` int(11) DEFAULT NULL,
  `laptop_name` varchar(255) DEFAULT NULL,
  `specs` text DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL,
  `booking_type` enum('recommendation','category','individual_laptop','standard') DEFAULT 'standard',
  `booking_status` enum('pending','confirmed','shipped','delivered','cancelled') DEFAULT 'pending',
  `booked_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `bookings`
--

INSERT INTO `bookings` (`bid`, `uid`, `rid`, `laptop_name`, `specs`, `price`, `booking_type`, `booking_status`, `booked_at`, `updated_at`) VALUES
(1, 2, 21, 'Asus Notebook', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 15930.72, 'recommendation', 'confirmed', '2025-10-31 10:26:04', '2025-10-31 10:48:10');

-- --------------------------------------------------------

--
-- Table structure for table `cluster_categories`
--

CREATE TABLE `cluster_categories` (
  `cid` int(11) NOT NULL,
  `uid` int(11) NOT NULL,
  `pid` int(11) DEFAULT NULL,
  `cluster_number` int(11) DEFAULT NULL,
  `cluster_name` varchar(255) DEFAULT NULL,
  `cluster_description` text DEFAULT NULL,
  `example_laptops` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`example_laptops`)),
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cluster_categories`
--

INSERT INTO `cluster_categories` (`cid`, `uid`, `pid`, `cluster_number`, `cluster_name`, `cluster_description`, `example_laptops`, `created_at`) VALUES
(2, 2, 2, 4, 'Entry-Level Basic Laptops', 'Laptops with similar specifications and price range (Cluster 4)', '[{\"Company\": \"HP\", \"TypeName\": \"2 in 1 Convertible\", \"Title\": \"HP 2 in 1 Convertible\", \"Ram\": \"8GB\", \"Storage\": \"64GB SSD\", \"Cpu_brand\": \"Other Intel Processor\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.4kg\", \"Price\": 26373.6, \"Features\": \"Touchscreen, Good Memory\", \"Touchscreen\": \"Yes\", \"Ips\": \"No\", \"os\": \"Others/No OS/Linux\"}, {\"Company\": \"Toshiba\", \"TypeName\": \"Notebook\", \"Title\": \"Toshiba Notebook\", \"Ram\": \"8GB\", \"Storage\": \"500GB HDD\", \"Cpu_brand\": \"Intel Core i5\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.0kg\", \"Price\": 56689.92, \"Features\": \"IPS Display, Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"Yes\", \"os\": \"Windows\"}, {\"Company\": \"Dell\", \"TypeName\": \"Notebook\", \"Title\": \"Dell Notebook\", \"Ram\": \"8GB\", \"Storage\": \"1000GB HDD\", \"Cpu_brand\": \"Intel Core i3\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.3kg\", \"Price\": 24988.85, \"Features\": \"Touchscreen, Good Memory\", \"Touchscreen\": \"Yes\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"HP\", \"TypeName\": \"Notebook\", \"Title\": \"HP Notebook\", \"Ram\": \"8GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"Intel Core i3\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.9kg\", \"Price\": 28717.92, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Lenovo\", \"TypeName\": \"Notebook\", \"Title\": \"Lenovo Notebook\", \"Ram\": \"4GB\", \"Storage\": \"32GB SSD\", \"Cpu_brand\": \"Other Intel Processor\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.4kg\", \"Price\": 14598.72, \"Features\": \"Standard Features\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}]', '2025-10-31 09:54:57'),
(3, 2, 3, 4, 'Entry-Level Basic Laptops', 'Laptops with similar specifications and price range (Cluster 4)', '[{\"Company\": \"HP\", \"TypeName\": \"Notebook\", \"Title\": \"HP Notebook\", \"Ram\": \"8GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"Intel Core i5\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.9kg\", \"Price\": 30849.12, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"HP\", \"TypeName\": \"Notebook\", \"Title\": \"HP Notebook\", \"Ram\": \"8GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"Intel Core i5\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.3kg\", \"Price\": 59886.72, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"HP\", \"TypeName\": \"Notebook\", \"Title\": \"HP Notebook\", \"Ram\": \"8GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"Intel Core i5\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.0kg\", \"Price\": 31914.19, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Dell\", \"TypeName\": \"Notebook\", \"Title\": \"Dell Notebook\", \"Ram\": \"8GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"Intel Core i5\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.2kg\", \"Price\": 39960.0, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Acer\", \"TypeName\": \"Notebook\", \"Title\": \"Acer Notebook\", \"Ram\": \"6GB\", \"Storage\": \"1000GB HDD\", \"Cpu_brand\": \"Intel Core i3\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.1kg\", \"Price\": 25521.12, \"Features\": \"Touchscreen, IPS Display\", \"Touchscreen\": \"Yes\", \"Ips\": \"Yes\", \"os\": \"Windows\"}]', '2025-10-31 10:21:25'),
(4, 2, 4, 2, 'Mid-Range Productivity Laptops', 'Laptops with similar specifications and price range (Cluster 2)', '[{\"Company\": \"Lenovo\", \"TypeName\": \"2 in 1 Convertible\", \"Title\": \"Lenovo 2 in 1 Convertible\", \"Ram\": \"16GB\", \"Storage\": \"512GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.4kg\", \"Price\": 93186.72, \"Features\": \"Touchscreen, High Memory\", \"Touchscreen\": \"Yes\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"HP\", \"TypeName\": \"Notebook\", \"Title\": \"HP Notebook\", \"Ram\": \"16GB\", \"Storage\": \"512GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.5kg\", \"Price\": 63776.16, \"Features\": \"High Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Toshiba\", \"TypeName\": \"Ultrabook\", \"Title\": \"Toshiba Ultrabook\", \"Ram\": \"16GB\", \"Storage\": \"512GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.2kg\", \"Price\": 99367.2, \"Features\": \"Touchscreen, High Memory\", \"Touchscreen\": \"Yes\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Samsung\", \"TypeName\": \"Ultrabook\", \"Title\": \"Samsung Ultrabook\", \"Ram\": \"16GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Nvidia\", \"Weight\": \"1.2kg\", \"Price\": 98514.72, \"Features\": \"High Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Toshiba\", \"TypeName\": \"Ultrabook\", \"Title\": \"Toshiba Ultrabook\", \"Ram\": \"16GB\", \"Storage\": \"512GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.4kg\", \"Price\": 100006.56, \"Features\": \"Touchscreen, High Memory\", \"Touchscreen\": \"Yes\", \"Ips\": \"No\", \"os\": \"Windows\"}]', '2025-10-31 10:22:03'),
(5, 2, 5, 4, 'Entry-Level Basic Laptops', 'Laptops with similar specifications and price range (Cluster 4)', '[{\"Company\": \"Toshiba\", \"TypeName\": \"Notebook\", \"Title\": \"Toshiba Notebook\", \"Ram\": \"8GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"Intel Core i5\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.2kg\", \"Price\": 59620.32, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"HP\", \"TypeName\": \"Notebook\", \"Title\": \"HP Notebook\", \"Ram\": \"8GB\", \"Storage\": \"1000GB HDD\", \"Cpu_brand\": \"Intel Core i3\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.6kg\", \"Price\": 28992.31, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Toshiba\", \"TypeName\": \"Notebook\", \"Title\": \"Toshiba Notebook\", \"Ram\": \"4GB\", \"Storage\": \"128GB SSD\", \"Cpu_brand\": \"Intel Core i3\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.1kg\", \"Price\": 26533.44, \"Features\": \"Standard Features\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Lenovo\", \"TypeName\": \"Notebook\", \"Title\": \"Lenovo Notebook\", \"Ram\": \"6GB\", \"Storage\": \"256GB SSD\", \"Cpu_brand\": \"AMD Processor\", \"Gpu_brand\": \"AMD\", \"Weight\": \"2.2kg\", \"Price\": 29250.72, \"Features\": \"Standard Features\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Acer\", \"TypeName\": \"Notebook\", \"Title\": \"Acer Notebook\", \"Ram\": \"8GB\", \"Storage\": \"1000GB HDD\", \"Cpu_brand\": \"Intel Core i3\", \"Gpu_brand\": \"Intel\", \"Weight\": \"2.1kg\", \"Price\": 20725.92, \"Features\": \"Good Memory\", \"Touchscreen\": \"No\", \"Ips\": \"No\", \"os\": \"Windows\"}]', '2025-10-31 10:22:34'),
(6, 2, 6, 0, 'Premium Ultrabooks', 'Laptops with similar specifications and price range (Cluster 0)', '[{\"Company\": \"Asus\", \"TypeName\": \"Gaming\", \"Title\": \"Asus Gaming\", \"Ram\": \"64GB\", \"Storage\": \"1000GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Nvidia\", \"Weight\": \"3.6kg\", \"Price\": 211788.0, \"Features\": \"IPS Display, High Memory\", \"Touchscreen\": \"No\", \"Ips\": \"Yes\", \"os\": \"Windows\"}, {\"Company\": \"Razer\", \"TypeName\": \"Gaming\", \"Title\": \"Razer Gaming\", \"Ram\": \"32GB\", \"Storage\": \"512GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Nvidia\", \"Weight\": \"3.5kg\", \"Price\": 292986.72, \"Features\": \"Touchscreen, High Memory\", \"Touchscreen\": \"Yes\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Toshiba\", \"TypeName\": \"Ultrabook\", \"Title\": \"Toshiba Ultrabook\", \"Ram\": \"32GB\", \"Storage\": \"512GB SSD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Intel\", \"Weight\": \"1.1kg\", \"Price\": 149130.72, \"Features\": \"Touchscreen, High Memory\", \"Touchscreen\": \"Yes\", \"Ips\": \"No\", \"os\": \"Windows\"}, {\"Company\": \"Lenovo\", \"TypeName\": \"Gaming\", \"Title\": \"Lenovo Gaming\", \"Ram\": \"32GB\", \"Storage\": \"256GB SSD + 1000GB HDD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Nvidia\", \"Weight\": \"4.6kg\", \"Price\": 141884.64, \"Features\": \"IPS Display, High Memory\", \"Touchscreen\": \"No\", \"Ips\": \"Yes\", \"os\": \"Windows\"}, {\"Company\": \"Dell\", \"TypeName\": \"Gaming\", \"Title\": \"Dell Gaming\", \"Ram\": \"32GB\", \"Storage\": \"512GB SSD + 1000GB HDD\", \"Cpu_brand\": \"Intel Core i7\", \"Gpu_brand\": \"Nvidia\", \"Weight\": \"4.4kg\", \"Price\": 163723.58, \"Features\": \"IPS Display, High Memory\", \"Touchscreen\": \"No\", \"Ips\": \"Yes\", \"os\": \"Windows\"}]', '2025-10-31 10:39:49');

-- --------------------------------------------------------

--
-- Table structure for table `predictions`
--

CREATE TABLE `predictions` (
  `pid` int(11) NOT NULL,
  `uid` int(11) NOT NULL,
  `company` varchar(50) DEFAULT NULL,
  `type` varchar(50) DEFAULT NULL,
  `ram` int(11) DEFAULT NULL,
  `weight` decimal(5,2) DEFAULT NULL,
  `touchscreen` tinyint(1) DEFAULT NULL,
  `ips` tinyint(1) DEFAULT NULL,
  `screen_size` decimal(4,2) DEFAULT NULL,
  `resolution` varchar(20) DEFAULT NULL,
  `cpu` varchar(100) DEFAULT NULL,
  `hdd` int(11) DEFAULT NULL,
  `ssd` int(11) DEFAULT NULL,
  `gpu` varchar(100) DEFAULT NULL,
  `os` varchar(50) DEFAULT NULL,
  `predicted_price` decimal(10,2) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `predictions`
--

INSERT INTO `predictions` (`pid`, `uid`, `company`, `type`, `ram`, `weight`, `touchscreen`, `ips`, `screen_size`, `resolution`, `cpu`, `hdd`, `ssd`, `gpu`, `os`, `predicted_price`, `created_at`) VALUES
(2, 2, 'Acer', '2 in 1 Convertible', 4, 2.00, 0, 0, 15.60, '1920x1080', 'AMD Processor', 0, 256, 'AMD', 'Mac', 42947.04, '2025-10-31 09:54:57'),
(3, 2, 'Acer', '2 in 1 Convertible', 4, 2.00, 0, 0, 15.60, '1920x1080', 'AMD Processor', 0, 512, 'AMD', 'Mac', 46020.07, '2025-10-31 10:21:25'),
(4, 2, 'Apple', 'Notebook', 8, 2.00, 1, 1, 14.00, '1920x1080', 'Intel Core i7', 0, 512, 'AMD', 'Mac', 64187.44, '2025-10-31 10:22:03'),
(5, 2, 'Acer', '2 in 1 Convertible', 4, 2.00, 0, 0, 15.60, '1920x1080', 'AMD Processor', 0, 256, 'AMD', 'Mac', 42947.04, '2025-10-31 10:22:34'),
(6, 2, 'Acer', '2 in 1 Convertible', 16, 2.00, 0, 0, 12.00, '1920x1080', 'AMD Processor', 0, 256, 'AMD', 'Windows', 51531.28, '2025-10-31 10:39:49'),
(7, 2, 'Apple', '2 in 1 Convertible', 4, 2.00, 0, 0, 12.00, '1920x1080', 'AMD Processor', 0, 128, 'AMD', 'Mac', 45223.61, '2025-10-31 10:42:32');

-- --------------------------------------------------------

--
-- Table structure for table `recommendations`
--

CREATE TABLE `recommendations` (
  `rid` int(11) NOT NULL,
  `uid` int(11) NOT NULL,
  `pid` int(11) DEFAULT NULL,
  `laptop_name` varchar(255) DEFAULT NULL,
  `specs` text DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL,
  `similarity_score` decimal(5,4) DEFAULT NULL,
  `saved_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `recommendations`
--

INSERT INTO `recommendations` (`rid`, `uid`, `pid`, `laptop_name`, `specs`, `price`, `similarity_score`, `saved_at`) VALUES
(6, 2, 2, 'Asus Notebook', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 15930.72, 0.5390, '2025-10-31 09:54:57'),
(7, 2, 2, 'HP Notebook', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i7, GPU: Intel', 69477.12, 0.4400, '2025-10-31 09:54:57'),
(8, 2, 2, 'Toshiba Notebook', 'RAM: 8GB, Storage: 500GB HDD, CPU: Intel Core i5, GPU: Intel', 56689.92, 0.4370, '2025-10-31 09:54:57'),
(9, 2, 2, 'MSI Gaming', 'RAM: 16GB, Storage: 256GB SSD + 1000GB HDD, CPU: Intel Core i7, GPU: Nvidia', 120831.58, 0.4350, '2025-10-31 09:54:57'),
(10, 2, 2, 'Toshiba Ultrabook', 'RAM: 16GB, Storage: 512GB SSD, CPU: Intel Core i7, GPU: Intel', 100006.56, 0.4240, '2025-10-31 09:54:57'),
(11, 2, 3, 'Asus Notebook', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 15930.72, 0.5040, '2025-10-31 10:21:25'),
(12, 2, 3, 'Dell Ultrabook', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i7, GPU: Intel', 77202.72, 0.4310, '2025-10-31 10:21:25'),
(13, 2, 3, 'HP Notebook', 'RAM: 4GB, Storage: 500GB HDD, CPU: Intel Core i5, GPU: Intel', 20986.99, 0.4310, '2025-10-31 10:21:25'),
(14, 2, 3, 'Acer Ultrabook', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i5, GPU: Intel', 41025.60, 0.4310, '2025-10-31 10:21:25'),
(15, 2, 3, 'Toshiba Notebook', 'RAM: 8GB, Storage: 500GB HDD, CPU: Intel Core i5, GPU: Intel', 56689.92, 0.4090, '2025-10-31 10:21:25'),
(16, 2, 4, 'Acer Notebook', 'RAM: 12GB, Storage: 1000GB HDD, CPU: Intel Core i7, GPU: Intel', 35111.52, 0.4170, '2025-10-31 10:22:03'),
(17, 2, 4, 'Dell Notebook', 'RAM: 8GB, Storage: 1000GB HDD, CPU: Intel Core i5, GPU: Intel', 50882.40, 0.4160, '2025-10-31 10:22:03'),
(18, 2, 4, 'Lenovo Gaming', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i5, GPU: Nvidia', 44169.12, 0.4130, '2025-10-31 10:22:03'),
(19, 2, 4, 'HP Notebook', 'RAM: 6GB, Storage: 256GB SSD, CPU: Intel Core i5, GPU: AMD', 32980.32, 0.4110, '2025-10-31 10:22:03'),
(20, 2, 4, 'Asus Notebook', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 15930.72, 0.3860, '2025-10-31 10:22:03'),
(21, 2, 5, 'Asus Notebook', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 15930.72, 0.5390, '2025-10-31 10:22:34'),
(22, 2, 5, 'HP Notebook', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i7, GPU: Intel', 69477.12, 0.4400, '2025-10-31 10:22:34'),
(23, 2, 5, 'Toshiba Notebook', 'RAM: 8GB, Storage: 500GB HDD, CPU: Intel Core i5, GPU: Intel', 56689.92, 0.4370, '2025-10-31 10:22:34'),
(24, 2, 5, 'MSI Gaming', 'RAM: 16GB, Storage: 256GB SSD + 1000GB HDD, CPU: Intel Core i7, GPU: Nvidia', 120831.58, 0.4350, '2025-10-31 10:22:34'),
(25, 2, 5, 'Toshiba Ultrabook', 'RAM: 16GB, Storage: 512GB SSD, CPU: Intel Core i7, GPU: Intel', 100006.56, 0.4240, '2025-10-31 10:22:34'),
(26, 2, 6, 'Asus 2 in 1 Convertible', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 19980.00, 0.5190, '2025-10-31 10:39:49'),
(27, 2, 6, 'Lenovo 2 in 1 Convertible', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i5, GPU: Intel', 53226.72, 0.5150, '2025-10-31 10:39:49'),
(28, 2, 6, 'Asus Notebook', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 15930.72, 0.5150, '2025-10-31 10:39:49'),
(29, 2, 6, 'Dell Notebook', 'RAM: 16GB, Storage: 1000GB SSD, CPU: Intel Core i7, GPU: Nvidia', 127818.72, 0.5120, '2025-10-31 10:39:49'),
(30, 2, 6, 'Razer Gaming', 'RAM: 32GB, Storage: 1000GB SSD, CPU: Intel Core i7, GPU: Nvidia', 324954.72, 0.5000, '2025-10-31 10:39:49'),
(31, 2, 7, 'Acer Notebook', 'RAM: 6GB, Storage: 1000GB HDD, CPU: Intel Core i5, GPU: Intel', 29250.72, 0.4240, '2025-10-31 10:42:32'),
(32, 2, 7, 'Lenovo Notebook', 'RAM: 4GB, Storage: 1000GB HDD, CPU: Intel Core i3, GPU: Intel', 24988.32, 0.4210, '2025-10-31 10:42:32'),
(33, 2, 7, 'Dell Notebook', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i5, GPU: Intel', 59513.23, 0.4030, '2025-10-31 10:42:32'),
(34, 2, 7, 'Asus Notebook', 'RAM: 4GB, Storage: 32GB SSD, CPU: Other Intel Processor, GPU: Intel', 15930.72, 0.3980, '2025-10-31 10:42:32'),
(35, 2, 7, 'Lenovo Ultrabook', 'RAM: 8GB, Storage: 256GB SSD, CPU: Intel Core i7, GPU: AMD', 64755.45, 0.3940, '2025-10-31 10:42:32');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `uid` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`uid`, `username`, `email`, `password`, `created_at`) VALUES
(2, 'ayush010', 'ayush@gmail.com', 'pbkdf2:sha256:600000$VzNGCAivsoWzVtgV$30d0b771154e983ab4259d3724e5c0126005133b820078bb57a4039c0e17d448', '2025-10-31 09:21:59');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `bookings`
--
ALTER TABLE `bookings`
  ADD PRIMARY KEY (`bid`),
  ADD KEY `rid` (`rid`),
  ADD KEY `idx_bookings_uid` (`uid`),
  ADD KEY `idx_bookings_status` (`booking_status`);

--
-- Indexes for table `cluster_categories`
--
ALTER TABLE `cluster_categories`
  ADD PRIMARY KEY (`cid`),
  ADD KEY `idx_cluster_categories_uid` (`uid`),
  ADD KEY `idx_cluster_categories_pid` (`pid`);

--
-- Indexes for table `predictions`
--
ALTER TABLE `predictions`
  ADD PRIMARY KEY (`pid`),
  ADD KEY `idx_predictions_uid` (`uid`),
  ADD KEY `idx_predictions_created_at` (`created_at`);

--
-- Indexes for table `recommendations`
--
ALTER TABLE `recommendations`
  ADD PRIMARY KEY (`rid`),
  ADD KEY `idx_recommendations_uid` (`uid`),
  ADD KEY `idx_recommendations_pid` (`pid`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`uid`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `bookings`
--
ALTER TABLE `bookings`
  MODIFY `bid` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `cluster_categories`
--
ALTER TABLE `cluster_categories`
  MODIFY `cid` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `predictions`
--
ALTER TABLE `predictions`
  MODIFY `pid` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `recommendations`
--
ALTER TABLE `recommendations`
  MODIFY `rid` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=36;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `uid` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `bookings`
--
ALTER TABLE `bookings`
  ADD CONSTRAINT `bookings_ibfk_1` FOREIGN KEY (`uid`) REFERENCES `users` (`uid`) ON DELETE CASCADE,
  ADD CONSTRAINT `bookings_ibfk_2` FOREIGN KEY (`rid`) REFERENCES `recommendations` (`rid`) ON DELETE SET NULL;

--
-- Constraints for table `cluster_categories`
--
ALTER TABLE `cluster_categories`
  ADD CONSTRAINT `cluster_categories_ibfk_1` FOREIGN KEY (`uid`) REFERENCES `users` (`uid`) ON DELETE CASCADE,
  ADD CONSTRAINT `cluster_categories_ibfk_2` FOREIGN KEY (`pid`) REFERENCES `predictions` (`pid`) ON DELETE CASCADE;

--
-- Constraints for table `predictions`
--
ALTER TABLE `predictions`
  ADD CONSTRAINT `predictions_ibfk_1` FOREIGN KEY (`uid`) REFERENCES `users` (`uid`) ON DELETE CASCADE;

--
-- Constraints for table `recommendations`
--
ALTER TABLE `recommendations`
  ADD CONSTRAINT `recommendations_ibfk_1` FOREIGN KEY (`uid`) REFERENCES `users` (`uid`) ON DELETE CASCADE,
  ADD CONSTRAINT `recommendations_ibfk_2` FOREIGN KEY (`pid`) REFERENCES `predictions` (`pid`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
