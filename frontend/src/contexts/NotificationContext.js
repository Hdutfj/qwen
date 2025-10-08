import React, { createContext, useContext, useReducer } from 'react';

const NotificationContext = createContext();

export const useNotifications = () => {
  return useContext(NotificationContext);
};

const notificationReducer = (state, action) => {
  switch (action.type) {
    case 'ADD_NOTIFICATION':
      return [...state, action.notification];
    case 'REMOVE_NOTIFICATION':
      return state.filter(notification => notification.id !== action.id);
    case 'CLEAR_NOTIFICATIONS':
      return [];
    default:
      return state;
  }
};

export const NotificationProvider = ({ children }) => {
  const [notifications, dispatch] = useReducer(notificationReducer, []);

  const addNotification = (type, message) => {
    const id = Date.now() + Math.random();
    dispatch({
      type: 'ADD_NOTIFICATION',
      notification: { id, type, message }
    });
  };

  const removeNotification = (id) => {
    dispatch({ type: 'REMOVE_NOTIFICATION', id });
  };

  const clearNotifications = () => {
    dispatch({ type: 'CLEAR_NOTIFICATIONS' });
  };

  const value = {
    notifications,
    addNotification,
    removeNotification,
    clearNotifications
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};