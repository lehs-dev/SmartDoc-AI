#!/bin/bash
rm db.sqlite3
echo "Đã xóa db.sqlite3"
rm -rf media/documents/*
echo "Đã xóa tất cả tệp trong media/documents/"
rm -rf vectors/*
echo "Đã xóa tất cả tệp trong vectors/"
echo "Đã hoàn thành việc reset cơ sở dữ liệu và xóa các tệp liên quan!"