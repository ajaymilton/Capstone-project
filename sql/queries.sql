-- Q1: WHERE filters rows first, HAVING filters the groups afterwards
SELECT customer_id,
       COUNT(*)          AS big_orders,
       SUM(order_amount) AS total_big_orders
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_big_orders DESC;

-- Q2: revenue by city and product category (three-table join)
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: products priced above the average catalogue price (scalar subquery)
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers with no completed order (LEFT JOIN, keep the unmatched rows)
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: customers whose completed spend is above the average customer spend (two CTEs)
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total FROM customer_spend
)
SELECT cs.customer_id,
       ROUND(cs.total_spend, 2) AS total_spend,
       ROUND(a.avg_total, 2)    AS avg_total
FROM customer_spend cs
CROSS JOIN avg_spend a
WHERE cs.total_spend > a.avg_total
ORDER BY cs.total_spend DESC;

-- Q6: spend ranking inside each city (RANK / DENSE_RANK / ROW_NUMBER)
WITH spend AS (
    SELECT o.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- Q7: daily completed revenue with a running total
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
