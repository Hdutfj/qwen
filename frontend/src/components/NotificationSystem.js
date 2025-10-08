import React, { useEffect } from 'react';
import { useNotifications } from '../contexts/NotificationContext';
import { CheckCircleFill, ExclamationCircleFill, InfoCircleFill } from 'react-bootstrap-icons';

const NotificationSystem = () => {
  const { notifications, removeNotification } = useNotifications();

  useEffect(() => {
    const timers = notifications.map(notification => {
      const timer = setTimeout(() => {
        removeNotification(notification.id);
      }, 5000);
      return timer;
    });

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [notifications, removeNotification]);

  return (
    <div className="notification-container">
      {notifications.map(notification => (
        <div 
          key={notification.id} 
          className={`notification ${notification.type}`}
        >
          <div className="icon">
            {notification.type === 'success' && <CheckCircleFill />}
            {notification.type === 'error' && <ExclamationCircleFill />}
            {notification.type === 'info' && <InfoCircleFill />}
          </div>
          <div className="message">{notification.message}</div>
          <button 
            className="close-btn" 
            onClick={() => removeNotification(notification.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
};

export default NotificationSystem;